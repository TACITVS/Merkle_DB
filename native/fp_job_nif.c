#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <erl_nif.h>

// KMeans types live in fp_kmeans.h (already included via generated_nif.c).
extern ErlNifResourceType* RES_TYPE_KMeansResult;

typedef enum {
    JOB_QUEUED = 0,
    JOB_RUNNING = 1,
    JOB_DONE = 2,
    JOB_ERROR = 3,
    JOB_CANCELLED = 4
} JobStatus;

typedef enum {
    JOB_PHASE_INIT = 0,
    JOB_PHASE_COMPUTE = 1,
    JOB_PHASE_FINALIZE = 2
} JobPhase;

typedef enum {
    JOB_OP_KMEANS = 1
} JobOp;

typedef struct {
    ErlNifMutex* lock;
    JobStatus status;
    JobPhase phase;
    JobOp op;
    uint64_t iter;
    uint64_t max_iter;
    uint64_t started_at_ms;
    uint64_t updated_at_ms;
    int cancel_requested;
    int cancelled_by_progress;
    char message[128];
    char error[128];

    // KMeans args
    ErlNifEnv* job_env;
    ERL_NIF_TERM data_term;
    size_t data_bytes;
    size_t n;
    size_t d;
    size_t k;
    size_t max_iter_arg;
    double tol;
    unsigned int seed;

    // KMeans result
    int has_kmeans_result;
    KMeansResult kmeans_result;
} FPJob;

typedef struct JobNode {
    FPJob* job;
    struct JobNode* next;
} JobNode;

static ErlNifResourceType* RES_TYPE_FPJob = NULL;
static ErlNifMutex* queue_mutex = NULL;
static ErlNifCond* queue_cond = NULL;
static JobNode* queue_head = NULL;
static JobNode* queue_tail = NULL;
static ErlNifTid* worker_threads = NULL;
static int worker_thread_count = 0;
static int stop_workers = 0;

static ERL_NIF_TERM ATOM_OK;
static ERL_NIF_TERM ATOM_ERROR;
static ERL_NIF_TERM ATOM_RUNNING;
static ERL_NIF_TERM ATOM_QUEUED;
static ERL_NIF_TERM ATOM_DONE;
static ERL_NIF_TERM ATOM_CANCELLED;
static ERL_NIF_TERM ATOM_CONSUMED;
static ERL_NIF_TERM ATOM_STATUS;
static ERL_NIF_TERM ATOM_PHASE;
static ERL_NIF_TERM ATOM_ITER;
static ERL_NIF_TERM ATOM_MAX_ITER;
static ERL_NIF_TERM ATOM_STARTED_AT_MS;
static ERL_NIF_TERM ATOM_UPDATED_AT_MS;
static ERL_NIF_TERM ATOM_ELAPSED_MS;
static ERL_NIF_TERM ATOM_MESSAGE;
static ERL_NIF_TERM ATOM_OP;
static ERL_NIF_TERM ATOM_INIT;
static ERL_NIF_TERM ATOM_COMPUTE;
static ERL_NIF_TERM ATOM_FINALIZE;
static ERL_NIF_TERM ATOM_KMEANS;

static uint64_t now_ms(void) {
    return (uint64_t)enif_monotonic_time(ERL_NIF_MSEC);
}

static void job_set_message(FPJob* job, const char* msg) {
    enif_mutex_lock(job->lock);
    strncpy(job->message, msg, sizeof(job->message) - 1);
    job->message[sizeof(job->message) - 1] = '\0';
    job->updated_at_ms = now_ms();
    enif_mutex_unlock(job->lock);
}

static void job_set_status(FPJob* job, JobStatus status) {
    enif_mutex_lock(job->lock);
    job->status = status;
    job->updated_at_ms = now_ms();
    enif_mutex_unlock(job->lock);
}

static void job_set_phase(FPJob* job, JobPhase phase) {
    enif_mutex_lock(job->lock);
    job->phase = phase;
    job->updated_at_ms = now_ms();
    enif_mutex_unlock(job->lock);
}

static void enqueue_job(FPJob* job) {
    JobNode* node = (JobNode*)enif_alloc(sizeof(JobNode));
    node->job = job;
    node->next = NULL;

    enif_mutex_lock(queue_mutex);
    if (queue_tail == NULL) {
        queue_head = node;
        queue_tail = node;
    } else {
        queue_tail->next = node;
        queue_tail = node;
    }
    enif_cond_signal(queue_cond);
    enif_mutex_unlock(queue_mutex);
}

static FPJob* dequeue_job(void) {
    enif_mutex_lock(queue_mutex);
    while (!stop_workers && queue_head == NULL) {
        enif_cond_wait(queue_cond, queue_mutex);
    }
    if (stop_workers) {
        enif_mutex_unlock(queue_mutex);
        return NULL;
    }

    JobNode* node = queue_head;
    queue_head = node->next;
    if (queue_head == NULL) {
        queue_tail = NULL;
    }
    enif_mutex_unlock(queue_mutex);

    FPJob* job = node->job;
    enif_free(node);
    return job;
}

static int remove_job_from_queue(FPJob* job) {
    int removed = 0;
    enif_mutex_lock(queue_mutex);
    JobNode* prev = NULL;
    JobNode* cur = queue_head;
    while (cur) {
        if (cur->job == job) {
            if (prev) {
                prev->next = cur->next;
            } else {
                queue_head = cur->next;
            }
            if (queue_tail == cur) {
                queue_tail = prev;
            }
            enif_free(cur);
            removed = 1;
            break;
        }
        prev = cur;
        cur = cur->next;
    }
    enif_mutex_unlock(queue_mutex);
    return removed;
}

static int parse_kmeans_args(ErlNifEnv* env, ERL_NIF_TERM list, FPJob* job) {
    ERL_NIF_TERM head, tail;
    ErlNifBinary bin;
    ErlNifUInt64 val_n, val_d, val_k, val_max_iter;
    double val_tol;
    unsigned int val_seed;

    if (!enif_get_list_cell(env, list, &head, &tail)) return 0;
    if (!enif_inspect_binary(env, head, &bin)) return 0;
    job->data_bytes = bin.size;
    job->data_term = enif_make_copy(job->job_env, head);

    list = tail;
    if (!enif_get_list_cell(env, list, &head, &tail)) return 0;
    if (!enif_get_uint64(env, head, &val_n)) return 0;

    list = tail;
    if (!enif_get_list_cell(env, list, &head, &tail)) return 0;
    if (!enif_get_uint64(env, head, &val_d)) return 0;

    list = tail;
    if (!enif_get_list_cell(env, list, &head, &tail)) return 0;
    if (!enif_get_uint64(env, head, &val_k)) return 0;

    list = tail;
    if (!enif_get_list_cell(env, list, &head, &tail)) return 0;
    if (!enif_get_uint64(env, head, &val_max_iter)) return 0;

    list = tail;
    if (!enif_get_list_cell(env, list, &head, &tail)) return 0;
    if (!enif_get_double(env, head, &val_tol)) return 0;

    list = tail;
    if (!enif_get_list_cell(env, list, &head, &tail)) return 0;
    if (!enif_get_uint(env, head, &val_seed)) return 0;

    if (!enif_is_empty_list(env, tail)) return 0;

    if (val_n == 0 || val_d == 0 || val_k == 0) return 0;

    uint64_t expected = (uint64_t)val_n * (uint64_t)val_d * (uint64_t)sizeof(double);
    if (expected != (uint64_t)job->data_bytes) return 0;

    job->n = (size_t)val_n;
    job->d = (size_t)val_d;
    job->k = (size_t)val_k;
    job->max_iter_arg = (size_t)val_max_iter;
    job->tol = val_tol;
    job->seed = val_seed;
    job->max_iter = (uint64_t)val_max_iter;
    return 1;
}

static int kmeans_progress_cb(void* user, uint64_t iter, uint64_t max_iter, const char* phase) {
    FPJob* job = (FPJob*)user;
    int cancel = 0;

    enif_mutex_lock(job->lock);
    job->iter = iter;
    job->max_iter = max_iter;

    if (phase && phase[0] != '\0') {
        if (strcmp(phase, "init") == 0) {
            job->phase = JOB_PHASE_INIT;
        } else if (strcmp(phase, "finalize") == 0) {
            job->phase = JOB_PHASE_FINALIZE;
        } else {
            job->phase = JOB_PHASE_COMPUTE;
        }

        strncpy(job->message, phase, sizeof(job->message) - 1);
        job->message[sizeof(job->message) - 1] = '\0';
    }

    if (job->cancel_requested) {
        job->cancelled_by_progress = 1;
        cancel = 1;
    }

    job->updated_at_ms = now_ms();
    enif_mutex_unlock(job->lock);

    return cancel ? 0 : 1;
}

static void run_kmeans(FPJob* job) {
    job_set_phase(job, JOB_PHASE_COMPUTE);
    job_set_message(job, "kmeans");

    ErlNifBinary bin;
    if (!enif_inspect_binary(job->job_env, job->data_term, &bin)) {
        enif_mutex_lock(job->lock);
        job->status = JOB_ERROR;
        strncpy(job->error, "failed to inspect data", sizeof(job->error) - 1);
        job->error[sizeof(job->error) - 1] = '\0';
        job->updated_at_ms = now_ms();
        enif_mutex_unlock(job->lock);
        return;
    }

    if (job->cancel_requested) {
        job_set_status(job, JOB_CANCELLED);
        job_set_message(job, "cancelled");
        return;
    }

    job->cancelled_by_progress = 0;
    fp_progress_t progress = {kmeans_progress_cb, job};
    KMeansResult res = fp_kmeans_f64_progress((double*)bin.data, job->n, job->d, job->k,
                                              job->max_iter_arg, job->tol, job->seed, progress);

    if (job->cancel_requested || job->cancelled_by_progress) {
        fp_kmeans_free(&res);
        job_set_status(job, JOB_CANCELLED);
        job_set_message(job, "cancelled");
        return;
    }

    enif_mutex_lock(job->lock);
    job->kmeans_result = res;
    job->has_kmeans_result = 1;
    job->iter = (uint64_t)res.iterations;
    job->status = JOB_DONE;
    job->phase = JOB_PHASE_FINALIZE;
    strncpy(job->message, "done", sizeof(job->message) - 1);
    job->message[sizeof(job->message) - 1] = '\0';
    job->updated_at_ms = now_ms();
    enif_mutex_unlock(job->lock);
}

static void* job_worker(void* arg) {
    (void)arg;
    while (1) {
        FPJob* job = dequeue_job();
        if (!job) break;

        job_set_status(job, JOB_RUNNING);
        job_set_phase(job, JOB_PHASE_INIT);
        job_set_message(job, "running");

        switch (job->op) {
            case JOB_OP_KMEANS:
                run_kmeans(job);
                break;
            default:
                job_set_status(job, JOB_ERROR);
                job_set_message(job, "unknown op");
                break;
        }

        if (job->job_env) {
            enif_free_env(job->job_env);
            job->job_env = NULL;
        }

        enif_release_resource(job);
    }
    return NULL;
}

static void fp_job_dtor(ErlNifEnv* env, void* obj) {
    (void)env;
    FPJob* job = (FPJob*)obj;
    if (job->has_kmeans_result) {
        fp_kmeans_free(&job->kmeans_result);
        job->has_kmeans_result = 0;
    }
    if (job->job_env) {
        enif_free_env(job->job_env);
        job->job_env = NULL;
    }
    if (job->lock) {
        enif_mutex_destroy(job->lock);
        job->lock = NULL;
    }
}

static int read_thread_count(void) {
    const char* env = getenv("MERKLEDB_JOB_THREADS");
    if (!env || env[0] == '\0') return 1;
    int val = atoi(env);
    return val > 0 ? val : 1;
}

static void init_atoms(ErlNifEnv* env) {
    ATOM_OK = enif_make_atom(env, "ok");
    ATOM_ERROR = enif_make_atom(env, "error");
    ATOM_RUNNING = enif_make_atom(env, "running");
    ATOM_QUEUED = enif_make_atom(env, "queued");
    ATOM_DONE = enif_make_atom(env, "done");
    ATOM_CANCELLED = enif_make_atom(env, "cancelled");
    ATOM_CONSUMED = enif_make_atom(env, "consumed");
    ATOM_STATUS = enif_make_atom(env, "status");
    ATOM_PHASE = enif_make_atom(env, "phase");
    ATOM_ITER = enif_make_atom(env, "iter");
    ATOM_MAX_ITER = enif_make_atom(env, "max_iter");
    ATOM_STARTED_AT_MS = enif_make_atom(env, "started_at_ms");
    ATOM_UPDATED_AT_MS = enif_make_atom(env, "updated_at_ms");
    ATOM_ELAPSED_MS = enif_make_atom(env, "elapsed_ms");
    ATOM_MESSAGE = enif_make_atom(env, "message");
    ATOM_OP = enif_make_atom(env, "op");
    ATOM_INIT = enif_make_atom(env, "init");
    ATOM_COMPUTE = enif_make_atom(env, "compute");
    ATOM_FINALIZE = enif_make_atom(env, "finalize");
    ATOM_KMEANS = enif_make_atom(env, "fp_kmeans_f64");
}

static ERL_NIF_TERM status_atom(JobStatus status) {
    switch (status) {
        case JOB_QUEUED: return ATOM_QUEUED;
        case JOB_RUNNING: return ATOM_RUNNING;
        case JOB_DONE: return ATOM_DONE;
        case JOB_ERROR: return ATOM_ERROR;
        case JOB_CANCELLED: return ATOM_CANCELLED;
        default: return ATOM_ERROR;
    }
}

static ERL_NIF_TERM phase_atom(JobPhase phase) {
    switch (phase) {
        case JOB_PHASE_INIT: return ATOM_INIT;
        case JOB_PHASE_COMPUTE: return ATOM_COMPUTE;
        case JOB_PHASE_FINALIZE: return ATOM_FINALIZE;
        default: return ATOM_INIT;
    }
}

static ERL_NIF_TERM nif_fp_job_start(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 3) return enif_make_badarg(env);

    char op_name[64];
    if (!enif_get_atom(env, argv[0], op_name, sizeof(op_name), ERL_NIF_LATIN1)) {
        ErlNifBinary bin;
        if (!enif_inspect_binary(env, argv[0], &bin) || bin.size >= sizeof(op_name)) {
            return enif_make_badarg(env);
        }
        memcpy(op_name, bin.data, bin.size);
        op_name[bin.size] = '\0';
    }

    ERL_NIF_TERM args = argv[1];
    FPJob* job = (FPJob*)enif_alloc_resource(RES_TYPE_FPJob, sizeof(FPJob));
    memset(job, 0, sizeof(FPJob));
    job->lock = enif_mutex_create("fp_job_lock");
    job->status = JOB_QUEUED;
    job->phase = JOB_PHASE_INIT;
    job->iter = 0;
    job->max_iter = 0;
    job->started_at_ms = now_ms();
    job->updated_at_ms = job->started_at_ms;
    strncpy(job->message, "queued", sizeof(job->message) - 1);
    job->message[sizeof(job->message) - 1] = '\0';

    job->job_env = enif_alloc_env();

    if (strcmp(op_name, "fp_kmeans_f64") == 0) {
        job->op = JOB_OP_KMEANS;
        if (!parse_kmeans_args(env, args, job)) {
            enif_release_resource(job);
            return enif_make_badarg(env);
        }
    } else {
        enif_release_resource(job);
        return enif_make_badarg(env);
    }

    enif_keep_resource(job);
    enqueue_job(job);

    ERL_NIF_TERM term = enif_make_resource(env, job);
    enif_release_resource(job);
    (void)argv[2]; // opts reserved for future use
    return term;
}

static ERL_NIF_TERM nif_fp_job_status(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 1) return enif_make_badarg(env);
    FPJob* job;
    if (!enif_get_resource(env, argv[0], RES_TYPE_FPJob, (void**)&job)) return enif_make_badarg(env);

    ERL_NIF_TERM keys[8];
    ERL_NIF_TERM vals[8];
    int idx = 0;

    enif_mutex_lock(job->lock);
    uint64_t started = job->started_at_ms;
    uint64_t updated = job->updated_at_ms;
    JobStatus status = job->status;
    JobPhase phase = job->phase;
    uint64_t iter = job->iter;
    uint64_t max_iter = job->max_iter;
    const char* msg = job->message;
    JobOp op = job->op;
    enif_mutex_unlock(job->lock);

    keys[idx] = ATOM_STATUS; vals[idx] = status_atom(status); idx++;
    keys[idx] = ATOM_PHASE; vals[idx] = phase_atom(phase); idx++;
    keys[idx] = ATOM_ITER; vals[idx] = enif_make_uint64(env, iter); idx++;
    keys[idx] = ATOM_MAX_ITER; vals[idx] = enif_make_uint64(env, max_iter); idx++;
    keys[idx] = ATOM_STARTED_AT_MS; vals[idx] = enif_make_uint64(env, started); idx++;
    keys[idx] = ATOM_UPDATED_AT_MS; vals[idx] = enif_make_uint64(env, updated); idx++;
    keys[idx] = ATOM_ELAPSED_MS; vals[idx] = enif_make_uint64(env, updated - started); idx++;
    keys[idx] = ATOM_MESSAGE; vals[idx] = enif_make_string(env, msg, ERL_NIF_LATIN1); idx++;

    ERL_NIF_TERM map;
    enif_make_map_from_arrays(env, keys, vals, idx, &map);

    ERL_NIF_TERM op_term = (op == JOB_OP_KMEANS) ? ATOM_KMEANS : enif_make_string(env, "unknown", ERL_NIF_LATIN1);
    ERL_NIF_TERM map2;
    enif_make_map_put(env, map, ATOM_OP, op_term, &map2);
    return map2;
}

static ERL_NIF_TERM nif_fp_job_result(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 1) return enif_make_badarg(env);
    FPJob* job;
    if (!enif_get_resource(env, argv[0], RES_TYPE_FPJob, (void**)&job)) return enif_make_badarg(env);

    enif_mutex_lock(job->lock);
    JobStatus status = job->status;
    int has_result = job->has_kmeans_result;
    enif_mutex_unlock(job->lock);

    if (status == JOB_RUNNING || status == JOB_QUEUED) {
        return enif_make_tuple2(env, ATOM_ERROR, ATOM_RUNNING);
    }
    if (status == JOB_CANCELLED) {
        return enif_make_tuple2(env, ATOM_ERROR, ATOM_CANCELLED);
    }
    if (status == JOB_ERROR) {
        enif_mutex_lock(job->lock);
        const char* err = job->error[0] != '\0' ? job->error : "job_error";
        enif_mutex_unlock(job->lock);
        return enif_make_tuple2(env, ATOM_ERROR, enif_make_string(env, err, ERL_NIF_LATIN1));
    }
    if (!has_result) {
        return enif_make_tuple2(env, ATOM_ERROR, ATOM_CONSUMED);
    }

    if (job->op == JOB_OP_KMEANS) {
        enif_mutex_lock(job->lock);
        KMeansResult res = job->kmeans_result;
        job->has_kmeans_result = 0;
        enif_mutex_unlock(job->lock);

        KMeansResult* res_ptr = enif_alloc_resource(RES_TYPE_KMeansResult, sizeof(KMeansResult));
        *res_ptr = res;
        ERL_NIF_TERM res_term = enif_make_resource(env, res_ptr);
        enif_release_resource(res_ptr);
        return enif_make_tuple2(env, ATOM_OK, res_term);
    }

    return enif_make_tuple2(env, ATOM_ERROR, enif_make_string(env, "unsupported", ERL_NIF_LATIN1));
}

static ERL_NIF_TERM nif_fp_job_cancel(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    if (argc != 1) return enif_make_badarg(env);
    FPJob* job;
    if (!enif_get_resource(env, argv[0], RES_TYPE_FPJob, (void**)&job)) return enif_make_badarg(env);

    enif_mutex_lock(job->lock);
    JobStatus status = job->status;
    enif_mutex_unlock(job->lock);

    if (status == JOB_QUEUED) {
        if (remove_job_from_queue(job)) {
            job_set_status(job, JOB_CANCELLED);
            job_set_message(job, "cancelled");
            enif_release_resource(job);
            return ATOM_OK;
        }
    }

    enif_mutex_lock(job->lock);
    job->cancel_requested = 1;
    strncpy(job->message, "cancel_requested", sizeof(job->message) - 1);
    job->message[sizeof(job->message) - 1] = '\0';
    job->updated_at_ms = now_ms();
    enif_mutex_unlock(job->lock);

    return ATOM_OK;
}

static int fp_job_load(ErlNifEnv* env) {
    RES_TYPE_FPJob = enif_open_resource_type(env, NULL, "FPJob", fp_job_dtor,
                                             ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);
    if (!RES_TYPE_FPJob) return -1;

    queue_mutex = enif_mutex_create("fp_job_queue");
    queue_cond = enif_cond_create("fp_job_queue");
    if (!queue_mutex || !queue_cond) {
        if (queue_mutex) enif_mutex_destroy(queue_mutex);
        if (queue_cond) enif_cond_destroy(queue_cond);
        queue_mutex = NULL;
        queue_cond = NULL;
        return -1;
    }
    init_atoms(env);

    worker_thread_count = read_thread_count();
    worker_threads = (ErlNifTid*)enif_alloc(sizeof(ErlNifTid) * worker_thread_count);
    if (!worker_threads) {
        enif_mutex_destroy(queue_mutex);
        enif_cond_destroy(queue_cond);
        queue_mutex = NULL;
        queue_cond = NULL;
        return -1;
    }
    for (int i = 0; i < worker_thread_count; i++) {
        if (enif_thread_create("fp_job_worker", &worker_threads[i], job_worker, NULL, NULL) != 0) {
            stop_workers = 1;
            for (int j = 0; j < i; j++) {
                enif_thread_join(worker_threads[j], NULL);
            }
            enif_free(worker_threads);
            worker_threads = NULL;
            enif_mutex_destroy(queue_mutex);
            enif_cond_destroy(queue_cond);
            queue_mutex = NULL;
            queue_cond = NULL;
            return -1;
        }
    }

    return 0;
}

static void fp_job_unload(ErlNifEnv* env) {
    (void)env;
    if (!queue_mutex || !queue_cond) return;

    enif_mutex_lock(queue_mutex);
    stop_workers = 1;
    enif_cond_broadcast(queue_cond);
    enif_mutex_unlock(queue_mutex);

    if (worker_threads) {
        for (int i = 0; i < worker_thread_count; i++) {
            enif_thread_join(worker_threads[i], NULL);
        }
        enif_free(worker_threads);
        worker_threads = NULL;
    }

    enif_mutex_destroy(queue_mutex);
    enif_cond_destroy(queue_cond);
    queue_mutex = NULL;
    queue_cond = NULL;
}
