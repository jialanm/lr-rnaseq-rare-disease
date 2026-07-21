FROM continuumio/miniconda3:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:$PATH"

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    unzip \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

# Install bioinformatics tools via conda
RUN conda install -c bioconda -c conda-forge \
    gatk4 \
    salmon \
    samtools \
    && conda clean -a

# Install Google Cloud SDK
RUN curl https://sdk.cloud.google.com | bash -s -- --disable-prompts
ENV PATH="/root/google-cloud-sdk/bin:$PATH"

# Verify installations
RUN gatk --version && \
    salmon --version && \
    samtools --version && \
    gcloud version && \
    gcloud storage --help > /dev/null

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]