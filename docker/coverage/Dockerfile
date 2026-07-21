FROM google/cloud-sdk:slim

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install pysam pandas numpy intervaltree

# Copy your Python script
COPY read_from_bam.py ./read_from_bam.py