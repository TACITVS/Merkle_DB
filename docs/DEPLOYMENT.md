# MerkleDb Deployment Guide

This guide covers deploying MerkleDb to production environments, with a focus on Windows native deployment.

---

## Deployment Options

| Option | Best For | Complexity |
|--------|----------|------------|
| Mix Release | Production Windows | Medium |
| Direct Mix | Development/Testing | Low |
| Windows Service | Always-on Production | Medium |

---

## Prerequisites

### Windows Requirements

- **Windows 10/11** or **Windows Server 2016+**
- **Erlang/OTP 25+** installed
- **Visual C++ Redistributable** (usually included with Erlang installer)
- **Administrator access** (for service installation)

### Verify Erlang Installation

```batch
erl -eval "erlang:display(erlang:system_info(otp_release)), halt()." -noshell
```

Should display `"25"` or higher.

---

## Option 1: Mix Release (Recommended)

Mix releases create a self-contained package that includes the Erlang runtime.

### Step 1: Configure Environment

```batch
:: Set production mode
set MIX_ENV=prod

:: Set required API key
set MERKLE_DB_API_KEY=your_secure_32_character_key_here

:: Optional: Configure data directory
set MERKLE_DB_DATA_DIR=C:\MerkleDb\data

:: Optional: Disable HTTPS if no certificates
set MERKLE_DB_ENABLE_HTTPS=false
```

### Step 2: Build the Release

```batch
:: Get dependencies
mix deps.get --only prod

:: Compile
mix compile

:: Build release
mix release
```

This creates a release in `_build\prod\rel\merkle_db\`.

### Step 3: Run the Release

```batch
:: Start the server
_build\prod\rel\merkle_db\bin\merkle_db.bat start

:: Or start in foreground (for testing)
_build\prod\rel\merkle_db\bin\merkle_db.bat start_iex
```

### Step 4: Verify Deployment

```batch
:: Check health
curl http://localhost:4000/health/ready
```

### Release Commands

| Command | Description |
|---------|-------------|
| `merkle_db.bat start` | Start in background |
| `merkle_db.bat start_iex` | Start with interactive shell |
| `merkle_db.bat stop` | Stop the server |
| `merkle_db.bat restart` | Restart the server |
| `merkle_db.bat pid` | Show process ID |
| `merkle_db.bat remote` | Connect to running node |

---

## Option 2: Windows Service Installation

For production deployments that need to survive reboots and run without user login.

### Using NSSM (Non-Sucking Service Manager)

NSSM is a reliable service wrapper for Windows.

#### Step 1: Download NSSM

Download from https://nssm.cc/download and extract to `C:\nssm\`.

#### Step 2: Create the Service

```batch
:: Open admin command prompt
:: Install the service
C:\nssm\nssm.exe install MerkleDb

:: This opens a GUI. Configure:
:: Path: C:\MerkleDb\_build\prod\rel\merkle_db\bin\merkle_db.bat
:: Startup directory: C:\MerkleDb\_build\prod\rel\merkle_db
:: Arguments: start

:: Or use command line:
C:\nssm\nssm.exe install MerkleDb "C:\MerkleDb\_build\prod\rel\merkle_db\bin\merkle_db.bat" start
```

#### Step 3: Configure Environment Variables

```batch
:: Set environment variables for the service
C:\nssm\nssm.exe set MerkleDb AppEnvironmentExtra ^
  MERKLE_DB_API_KEY=your_key ^
  MERKLE_DB_DATA_DIR=C:\MerkleDb\data ^
  MERKLE_DB_ENABLE_HTTPS=false
```

#### Step 4: Configure Service Recovery

```batch
:: Restart on failure
C:\nssm\nssm.exe set MerkleDb AppExit Default Restart
C:\nssm\nssm.exe set MerkleDb AppRestartDelay 5000
```

#### Step 5: Start the Service

```batch
:: Start
net start MerkleDb

:: Or via NSSM
C:\nssm\nssm.exe start MerkleDb
```

### Service Management

```batch
:: Check status
sc query MerkleDb

:: Stop service
net stop MerkleDb

:: Restart service
net stop MerkleDb && net start MerkleDb

:: Remove service
C:\nssm\nssm.exe remove MerkleDb confirm
```

---

## Data Directory Structure

```
C:\MerkleDb\
├── data\
│   ├── wal\              # Write-Ahead Log
│   │   └── wal_*.bin
│   ├── snapshots\        # Checkpoints
│   │   └── checkpoint_*.bin
│   └── raft\             # Raft consensus data
│       └── nonode@nohost\
├── logs\                 # Application logs
│   └── merkle_db.log
└── _build\prod\rel\      # Release files
    └── merkle_db\
```

### Required Permissions

The service account needs:
- **Read/Write** access to `data\` directory
- **Read** access to release files
- **Network** access to configured ports

---

## HTTPS Configuration

### Generate Self-Signed Certificates

For testing or internal use:

```batch
:: Using OpenSSL (install via chocolatey or download)
openssl req -x509 -newkey rsa:4096 ^
  -keyout C:\MerkleDb\ssl\key.pem ^
  -out C:\MerkleDb\ssl\cert.pem ^
  -days 365 -nodes ^
  -subj "/CN=localhost"
```

### Configure HTTPS

```batch
set MERKLE_DB_ENABLE_HTTPS=true
set MERKLE_DB_SSL_CERT=C:\MerkleDb\ssl\cert.pem
set MERKLE_DB_SSL_KEY=C:\MerkleDb\ssl\key.pem
set MERKLE_DB_HTTPS_PORT=443
```

### Production Certificates

For production, obtain certificates from a CA:

1. Generate a CSR (Certificate Signing Request)
2. Submit to CA (Let's Encrypt, DigiCert, etc.)
3. Install the certificate and private key
4. Update paths in environment variables

---

## Firewall Configuration

### Allow HTTP/HTTPS Ports

```batch
:: Allow HTTP (admin command prompt)
netsh advfirewall firewall add rule ^
  name="MerkleDb HTTP" ^
  dir=in action=allow protocol=tcp localport=4000

:: Allow HTTPS
netsh advfirewall firewall add rule ^
  name="MerkleDb HTTPS" ^
  dir=in action=allow protocol=tcp localport=4443
```

### Remove Rules

```batch
netsh advfirewall firewall delete rule name="MerkleDb HTTP"
netsh advfirewall firewall delete rule name="MerkleDb HTTPS"
```

---

## Logging

### Configure Log Level

```batch
set MERKLE_DB_LOG_LEVEL=info
```

Available levels: `debug`, `info`, `warning`, `error`

### Log Rotation

For Windows, use a scheduled task to rotate logs:

```batch
:: Create rotation script (rotate_logs.bat)
@echo off
set LOGDIR=C:\MerkleDb\logs
set MAXSIZE=10485760

for %%F in (%LOGDIR%\*.log) do (
  for /f "tokens=3" %%S in ('dir /a-d "%%F" ^| find "%%~nxF"') do (
    if %%S GTR %MAXSIZE% (
      move "%%F" "%%F.%date:~-4%%date:~3,2%%date:~0,2%"
    )
  )
)
```

Schedule with Task Scheduler to run daily.

---

## Monitoring

### Health Check Script

Create `check_health.bat`:

```batch
@echo off
curl -s http://localhost:4000/health/ready > nul
if %errorlevel% neq 0 (
  echo MerkleDb is DOWN
  :: Send alert, restart service, etc.
  net stop MerkleDb
  timeout /t 5
  net start MerkleDb
) else (
  echo MerkleDb is UP
)
```

### Windows Event Log Integration

Add to your application config to send logs to Windows Event Log:

```elixir
# In config/prod.exs
config :logger,
  backends: [:console, {LoggerEventLog, :event_log}]

config :logger, :event_log,
  level: :info,
  source: "MerkleDb"
```

---

## Performance Tuning

### Erlang VM Flags

The release includes optimized VM flags in `rel/vm.args.eex`:

```
## Scheduler configuration
+S 4:4                      # 4 schedulers, 4 online
+SDcpu 2:2                  # Dirty CPU schedulers

## Memory allocation
+MBas aobf                  # Memory allocation strategy
+MBlmbcs 512               # Large block carrier size
+MBsmbcs 256               # Small block carrier size

## IO configuration
+A 64                       # Async thread pool size

## Distribution
+K true                     # Kernel polling
```

### Windows-Specific Optimizations

```batch
:: Increase file handle limit (if needed)
:: This requires registry modification and reboot

:: Disable ASLR for Erlang (may improve performance)
:: Not recommended for security-sensitive deployments
```

---

## Backup and Recovery

### Automated Backups

Create `backup.bat`:

```batch
@echo off
set BACKUP_DIR=C:\MerkleDb\backups
set DATA_DIR=C:\MerkleDb\data
set DATE=%date:~-4%%date:~3,2%%date:~0,2%

:: Create backup directory
mkdir %BACKUP_DIR%\%DATE% 2>nul

:: Stop accepting writes (optional - create checkpoint first)
curl -X POST http://localhost:4000/v1/default/checkpoint

:: Copy data
xcopy /E /Y %DATA_DIR%\* %BACKUP_DIR%\%DATE%\

:: Keep last 7 days of backups
forfiles /p %BACKUP_DIR% /d -7 /c "cmd /c rmdir /s /q @path" 2>nul

echo Backup completed: %BACKUP_DIR%\%DATE%
```

### Recovery Procedure

1. Stop the MerkleDb service
2. Remove or rename current data directory
3. Copy backup data to data directory
4. Start the MerkleDb service
5. Verify with health check

```batch
net stop MerkleDb
ren C:\MerkleDb\data C:\MerkleDb\data.old
xcopy /E /Y C:\MerkleDb\backups\20240108\* C:\MerkleDb\data\
net start MerkleDb
curl http://localhost:4000/health/ready
```

---

## Graceful Shutdown

MerkleDb handles shutdown gracefully when receiving a stop signal:

1. **Stops accepting new connections**
2. **Drains existing requests** (5 second timeout)
3. **Flushes Write-Ahead Log**
4. **Saves snapshot to disk**
5. **Leaves Raft cluster** (if clustered)
6. **Exits cleanly**

### Manual Graceful Shutdown

```batch
:: Via release command
_build\prod\rel\merkle_db\bin\merkle_db.bat stop

:: Via service
net stop MerkleDb
```

### Emergency Shutdown

If graceful shutdown hangs:

```batch
:: Find and kill the process
taskkill /F /IM erl.exe
```

**Warning:** Emergency shutdown may lose recent uncommitted data.

---

## Troubleshooting

### Server Won't Start

1. **Check port availability:**
   ```batch
   netstat -an | findstr :4000
   ```

2. **Check Erlang installation:**
   ```batch
   erl -eval "halt()."
   ```

3. **Check logs:**
   ```batch
   type _build\prod\rel\merkle_db\log\erlang.log.1
   ```

### Connection Refused

1. **Verify server is running:**
   ```batch
   tasklist | findstr beam
   ```

2. **Check firewall:**
   ```batch
   netsh advfirewall firewall show rule name="MerkleDb HTTP"
   ```

3. **Test locally:**
   ```batch
   curl http://127.0.0.1:4000/health/live
   ```

### High Memory Usage

1. **Reduce cache size:**
   ```batch
   set MERKLE_DB_CACHE_SIZE=10000
   ```

2. **Check for memory leaks:**
   ```batch
   curl http://localhost:4000/health/detailed
   ```
   Look at `system_metrics.memory_mb`

### Slow Queries

1. **Build an index** for large collections
2. **Check cache hit rate** via `/health/detailed`
3. **Increase cache size** if hit rate is low
4. **Use Int8 quantization** for memory efficiency

---

## Security Checklist

Before going to production:

- [ ] Strong API key set (32+ characters)
- [ ] HTTPS enabled with valid certificates
- [ ] Firewall configured to allow only necessary ports
- [ ] Data directory permissions restricted
- [ ] Log level set to `info` or `warning`
- [ ] Backup schedule configured
- [ ] Health monitoring in place
- [ ] Rate limiting configured appropriately

---

## Multi-Node Deployment

For high availability, deploy multiple MerkleDb nodes in a Raft cluster.

### Node 1 (First Node)

```batch
set MERKLE_DB_API_KEY=your_key
set MERKLE_DB_DATA_DIR=C:\MerkleDb\node1\data
mix run --no-halt
```

### Node 2 (Join Cluster)

```batch
set MERKLE_DB_API_KEY=your_key
set MERKLE_DB_DATA_DIR=C:\MerkleDb\node2\data

:: Start node
mix run --no-halt

:: Join cluster (in IEx)
MerkleDb.Raft.join_cluster(:"node1@hostname")
```

### Load Balancing

Use a reverse proxy (nginx, HAProxy) or Windows Network Load Balancing to distribute traffic across nodes.

---

## See Also

- [Quick Start Guide](QUICKSTART.md)
- [Configuration Guide](CONFIGURATION.md)
- [API Reference](API.md)
