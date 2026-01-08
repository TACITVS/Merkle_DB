# Changelog

All notable changes to MerkleDb are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Comprehensive documentation suite (QUICKSTART, CONFIGURATION, API, DEPLOYMENT)
- Python SDK with full API coverage

### Changed
- Improved README with production-ready examples

---

## [0.3.0] - 2024-01-08

### Added

#### Security
- **API Key Authentication**: Multi-key support with scopes (read, write, admin)
- **Rate Limiting**: Token bucket algorithm with per-IP and per-key limits
- **HTTPS Support**: Built-in TLS via Cowboy with configurable certificates
- **Input Validation**: Comprehensive validation for all API inputs
- **Protected Endpoints**: Global authentication middleware for `/v1/*` routes

#### Configuration
- **Runtime Configuration**: Environment variable support via `config/runtime.exs`
- **Config Validation**: Startup validation prevents misconfiguration
- **Configurable Settings**: All hardcoded values now configurable
  - Cache size and TTL
  - Rate limits
  - Raft timeouts
  - Query timeouts
  - Telemetry window size

#### Operations
- **Mix Releases**: Windows-native release support with batch scripts
- **Health Endpoints**: Kubernetes-compatible liveness and readiness probes
  - `GET /health/live` - Process running check
  - `GET /health/ready` - Service ready check
  - `GET /health/detailed` - Full metrics
- **Graceful Shutdown**: Proper drain, WAL flush, and snapshot on stop

#### Code Quality
- **Dialyzer Support**: Static type analysis
- **Credo Integration**: Code style and consistency checks
- **Mix Aliases**: `mix quality`, `mix test.all`, `mix release.build`

### Changed
- `lib/merkle_db/web/auth.ex` - Complete rewrite with ApiKeyStore integration
- `lib/merkle_db/web/router.ex` - Added global auth, rate limiting, health checks
- `lib/merkle_db/application.ex` - HTTPS, graceful shutdown, config validation
- `lib/merkle_db/raft.ex` - Configurable timeouts
- `lib/merkle_db/vector_cache.ex` - Configurable cache size and TTL
- `lib/merkle_db/telemetry_aggregator.ex` - Configurable window size

### New Files
- `lib/merkle_db/api_key_store.ex` - ETS-based API key management
- `lib/merkle_db/rate_limiter.ex` - Token bucket rate limiter
- `lib/merkle_db/validator.ex` - Input validation
- `lib/merkle_db/config_validator.ex` - Startup config validation
- `config/config.exs` - Base configuration
- `config/dev.exs` - Development overrides
- `config/test.exs` - Test environment
- `config/prod.exs` - Production defaults
- `config/runtime.exs` - Environment variable loading
- `rel/env.bat.eex` - Windows release environment
- `rel/vm.args.eex` - Erlang VM arguments
- `.credo.exs` - Credo configuration
- `.dialyzer_ignore.exs` - Dialyzer ignore patterns

---

## [0.2.0] - 2024-01-07

### Fixed

#### Stub Implementations
- **F32 Indexed Search**: Implemented `search_knn_ivf_f32` in `MerkleDb.NIF`
- **Bitmap Optimization**: Completed `optimize_bitmap/1` for inverted index compression

#### Bug Fixes
- **Logger.warn Deprecation**: Updated all `Logger.warn` calls to `Logger.warning`
- **Reset Collection**: Fixed collection reset to properly reinitialize tree state
- **ETS Race Condition**: Fixed `ConcurrentAccess` module ETS table initialization

#### Memory Safety (NIF)
- Fixed buffer overflow in HNSW distance calculation
- Added bounds checking for vector operations
- Fixed memory corruption in quantization routines
- Added proper cleanup in error paths

#### Raft Consensus
- Fixed hang in `MerkleDb.Raft.process_command`
- Added retry logic for leader discovery
- Fixed log replication edge cases

### Changed
- Improved error messages across NIF boundary
- Enhanced telemetry for debugging

---

## [0.1.0] - 2024-01-01

### Added

#### Core Features
- **Vector Storage**: High-performance KV store for vectors
- **Multiple Precisions**: Support for f64, f32, and int8 vectors
- **Similarity Search**: Brute-force and indexed KNN search
- **Metadata Filtering**: Rich filter expressions ($eq, $gt, $in, etc.)

#### Native Acceleration
- **AVX2 SIMD**: 256-bit vector operations in x86-64 assembly
- **Zero-Copy NIF**: Direct binary access without copying
- **Dirty Schedulers**: Heavy operations offloaded from main schedulers

#### Indexing
- **IVF Index**: Inverted File index for approximate search
- **K-Means Clustering**: Native implementation for index training
- **Int8 Quantization**: 4x memory reduction with scalar quantization

#### Distribution
- **Raft Consensus**: Strong consistency via Ra library
- **Leader Election**: Automatic failover
- **Log Replication**: Durable state machine

#### Persistence
- **Write-Ahead Log**: Durability for all writes
- **Snapshots**: Point-in-time checkpoints
- **Recovery**: Automatic state recovery on restart

#### API
- **HTTP REST API**: Full CRUD operations via Cowboy
- **Batch Operations**: Efficient bulk insert
- **Semantic Search**: Text embedding integration

#### Observability
- **Telemetry Integration**: Metrics emission
- **Dashboard**: Real-time metrics aggregation
- **Progress Tracking**: Index build progress

### Technical Details
- Built on Elixir/OTP for fault tolerance
- C and Assembly for performance-critical paths
- Ra library for Raft implementation
- Cowboy for HTTP serving

---

## Migration Guide

### From 0.2.x to 0.3.x

1. **API Keys Required in Production**
   ```bash
   # Set before starting
   set MERKLE_DB_API_KEY=your_secure_key
   set MIX_ENV=prod
   ```

2. **Update API Calls**
   ```bash
   # Add authentication header
   curl -H "Authorization: Bearer YOUR_KEY" http://localhost:4000/v1/collections
   ```

3. **Health Check URLs Changed**
   - Old: `/health`
   - New: `/health/live`, `/health/ready`, `/health/detailed`

4. **Configuration Files**
   - Move custom settings from hardcoded values to `config/runtime.exs`
   - Use environment variables for production configuration

### From 0.1.x to 0.2.x

1. **No Breaking Changes**
   - Bug fixes and stub implementations only
   - Existing API remains compatible

2. **Recommended Updates**
   - Update `Logger.warn` calls to `Logger.warning` in custom code
   - Review NIF error handling for improved error messages

---

## Roadmap

### v0.4.0 (Planned)
- [ ] HNSW index support
- [ ] Streaming vector ingestion
- [ ] Multi-collection transactions
- [ ] Backup/restore CLI commands

### v0.5.0 (Planned)
- [ ] Horizontal sharding
- [ ] Query result caching
- [ ] Async index building
- [ ] Prometheus metrics endpoint

### v1.0.0 (Future)
- [ ] Stable API guarantee
- [ ] Long-term support
- [ ] Enterprise features
