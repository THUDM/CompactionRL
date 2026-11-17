FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends tmux asciinema && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN python -c "from pathlib import Path; Path('ledger.txt').write_text(''.join(f'{i:04d}: record_{i}_payload checksum pending verification\\n' for i in range(300)))"
