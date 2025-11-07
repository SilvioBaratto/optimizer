# Portfolio Optimizer - Docker Scheduler

This Docker setup runs the portfolio optimizer's daily analysis tasks automatically at 22:00 Italy time (Europe/Rome timezone).

## Overview

The container runs two analysis scripts in sequence every day at 22:00 Italy time:

1. **Macro Regime Analysis** (`src/macro_regime/run_regime_analysis.py`)
   - Fetches economic indicators (VIX, HY spreads, PMI, yield curves)
   - Classifies business cycle regime for portfolio countries
   - Saves regime assessments to database

2. **Stock Signal Analysis** (`src/stock_analyzer/run_signal_analysis.py`)
   - Analyzes all active instruments in the universe
   - Generates daily signals using mathematical formulas
   - Applies cross-sectional standardization
   - Saves signals to database

## Quick Start

### Prerequisites

- Docker installed (v20.10+)
- Docker Compose installed (v2.0+)
- `.env` file configured in parent directory (`optimizer/.env`)

### Required Environment Variables

Ensure your `.env` file contains:

```bash
# Database (Supabase)
SUPABASE_DB_URL=postgresql+psycopg2://user:pass@host:6543/postgres
SUPABASE_URL=https://yourproject.supabase.co
SUPABASE_KEY=your-anon-key

# AI/LLM
OPENAI_API_KEY=your-openai-api-key

# Environment
ENVIRONMENT=production

# Optional
DEBUG=False

# PIA SOCKS5 Proxies (Optional - for Yahoo Finance rate limit protection)
# Format: user:pass@host:port,user:pass@host:port
PIA_SOCKS5_PROXIES=x4639735:6Etp5quBGv@proxy-nl.privateinternetaccess.com:1080,x4639735:6Etp5quBGv@proxy-uk.privateinternetaccess.com:1080
```

**Note on DNS Resolution**: The docker-compose.yml is configured with public DNS servers (Google 8.8.8.8, Cloudflare 1.1.1.1) to ensure PIA proxy hostnames can be resolved inside the container.

### Build and Run

```bash
# Navigate to docker directory
cd optimizer/docker

# Build and start container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop container
docker-compose down
```

## File Structure

```
optimizer/docker/
├── Dockerfile              # Container definition
├── docker-compose.yml      # Orchestration configuration
├── entrypoint.sh          # Container startup script
├── run_daily_analysis.sh  # Daily analysis execution script
├── crontab                # Cron schedule configuration
├── logs/                  # Log files (created on first run)
└── README.md              # This file
```

## Configuration

### Changing the Schedule

The container is configured to run at 22:00 Italy time (Europe/Rome timezone).

Edit `crontab` to change the execution time:

```bash
# Current: runs at 22:00 Italy time (Europe/Rome timezone)
0 22 * * * /app/run_daily_analysis.sh >> /var/log/cron.log 2>&1

# Example: run at 23:30 Italy time
30 23 * * * /app/run_daily_analysis.sh >> /var/log/cron.log 2>&1

# Example: run every 6 hours
0 */6 * * * /app/run_daily_analysis.sh >> /var/log/cron.log 2>&1
```

Rebuild the container after changing the schedule:

```bash
docker-compose down
docker-compose up -d --build
```

### Timezone Configuration

The container runs in Europe/Rome timezone (Italy) by default. To use a different timezone, edit `docker-compose.yml`:

```yaml
environment:
  - TZ=America/New_York  # Change to your timezone
```

**Note:** If you change the timezone, you'll also need to adjust the crontab schedule accordingly to maintain the desired local execution time.

### Run on Startup

To test the analysis immediately when the container starts:

```yaml
environment:
  - RUN_ON_STARTUP=true
```

## Monitoring

### View Container Logs

```bash
# Follow all logs
docker-compose logs -f

# View last 100 lines
docker-compose logs --tail=100

# View specific service logs
docker logs portfolio-optimizer-scheduler
```

### Check Cron Logs

```bash
# Access container shell
docker exec -it portfolio-optimizer-scheduler bash

# View cron logs
tail -f /var/log/cron.log

# View daily analysis logs
tail -f /var/log/optimizer/daily_analysis.log
```

### Health Check

```bash
# Check container status
docker-compose ps

# Manual health check
docker exec portfolio-optimizer-scheduler pgrep -f cron
```

## Troubleshooting

### Container exits immediately

**Check logs:**
```bash
docker-compose logs
```

**Common issues:**
- Missing environment variables (SUPABASE_DB_URL, OPENAI_API_KEY)
- Invalid `.env` file path
- Permissions issues with scripts

**Solution:**
```bash
# Verify .env exists
ls -la ../.env

# Check environment variables inside container
docker exec portfolio-optimizer-scheduler printenv
```

### Analysis scripts fail

**Check analysis logs:**
```bash
docker exec -it portfolio-optimizer-scheduler tail -n 100 /var/log/optimizer/daily_analysis.log
```

**Common issues:**
- Database connection failures (check SUPABASE_DB_URL)
- Missing API keys (OPENAI_API_KEY, FRED_API_KEY)
- Insufficient data in database

**Test scripts manually:**
```bash
# Access container shell
docker exec -it portfolio-optimizer-scheduler bash

# Run macro regime analysis manually
python3 src/macro_regime/run_regime_analysis.py

# Run stock signal analysis manually
python3 src/stock_analyzer/run_signal_analysis.py
```

### Cron jobs not running

**Verify cron is running:**
```bash
docker exec portfolio-optimizer-scheduler pgrep -f cron
```

**Check crontab:**
```bash
docker exec portfolio-optimizer-scheduler crontab -l
```

**Force cron to reload:**
```bash
docker exec portfolio-optimizer-scheduler crontab /etc/cron.d/optimizer-cron
docker restart portfolio-optimizer-scheduler
```

### Time synchronization issues

**Check container time:**
```bash
docker exec portfolio-optimizer-scheduler date
```

**Ensure timezone is correct:**
```bash
docker exec portfolio-optimizer-scheduler cat /etc/timezone
```

## Manual Execution

To run the analysis manually (outside the scheduled time):

```bash
# Option 1: Execute from host
docker exec portfolio-optimizer-scheduler /app/run_daily_analysis.sh

# Option 2: Execute from container shell
docker exec -it portfolio-optimizer-scheduler bash
/app/run_daily_analysis.sh
```

## Resource Management

### Default Resource Limits

- CPU: 2 cores max, 0.5 cores reserved
- Memory: 4GB max, 1GB reserved

### Adjust Limits

Edit `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'      # Increase for faster processing
      memory: 8G
    reservations:
      cpus: '1.0'
      memory: 2G
```

### Monitor Resource Usage

```bash
# Real-time stats
docker stats portfolio-optimizer-scheduler

# Container processes
docker top portfolio-optimizer-scheduler
```

## Persistence

### Log Persistence

Logs are persisted to `./logs/` directory (mounted as volume):

```bash
# View logs from host
tail -f docker/logs/daily_analysis.log
```

### Data Persistence

If you need to persist data files, uncomment in `docker-compose.yml`:

```yaml
volumes:
  - ../data:/app/data
```

## Production Deployment

### Best Practices

1. **Use secrets for sensitive data:**
   ```bash
   docker secret create supabase_db_url supabase_url.txt
   ```

2. **Set restart policy:**
   ```yaml
   restart: unless-stopped  # Already configured
   ```

3. **Enable log rotation:**
   ```yaml
   logging:
     options:
       max-size: "10m"
       max-file: "3"
   ```

4. **Monitor container health:**
   - Set up alerts for container restarts
   - Monitor log files for errors
   - Track database connection health

5. **Backup logs regularly:**
   ```bash
   # Create cron job on host to backup logs
   0 0 * * * tar -czf /backups/optimizer-logs-$(date +\%Y\%m\%d).tar.gz /path/to/optimizer/docker/logs/
   ```

## Uninstallation

```bash
# Stop and remove container
docker-compose down

# Remove container and volumes
docker-compose down -v

# Remove Docker image
docker rmi optimizer-scheduler
```

## Support

For issues or questions:
1. Check logs: `/var/log/optimizer/daily_analysis.log`
2. Review cron logs: `/var/log/cron.log`
3. Test scripts manually (see Troubleshooting section)
4. Verify environment variables are set correctly
5. Check database connectivity

## License

Same as parent project.
