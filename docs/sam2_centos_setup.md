# Setting up SAM2 on CentOS Server

## Important Note on CPU Usage

SAM2 (Segment Anything Model 2) cannot be run locally on CPU for several reasons:

1. **Architecture Requirements**
   - SAM2 is designed specifically for GPU acceleration
   - The model's architecture relies heavily on parallel processing capabilities of GPUs
   - The computational demands are too intensive for CPU-only processing

2. **Memory Requirements**
   - SAM2 requires significant memory bandwidth
   - GPUs provide the necessary high-speed memory access and parallel processing
   - CPU memory bandwidth would be insufficient for real-time or interactive use

3. **Performance Implications**
   - Running on CPU would make the interactive annotation process impractical
   - Response times would be too slow for real-time segmentation
   - Model inference would take several seconds to minutes per interaction

### Alternatives for CPU-Only Systems

If you don't have access to a GPU, consider these alternatives:

1. **Original SAM with ONNX Optimization**
   - Lighter weight than SAM2
   - Can run on CPU with reasonable performance
   - Some features may be limited

2. **MobileSAM**
   - Specifically designed for mobile and CPU deployment
   - Much faster inference times
   - Lower memory requirements
   - Suitable for laptop/desktop CPU usage

3. **Cloud-Based Solutions**
   - Use cloud GPU services (AWS, Google Cloud, Azure)
   - Set up SAM2 on a cloud instance
   - Access via API endpoints

4. **Simplified Annotation Tools**
   - Traditional computer vision approaches
   - Semi-automated tools that work well on CPU
   - Manual annotation tools with basic assistance features

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with CUDA support
- Minimum 8GB GPU memory recommended
- CPU: 8+ cores recommended
- RAM: 16GB minimum, 32GB recommended
- Storage: 20GB+ free space

### Software Requirements
- CentOS 7 or 8
- NVIDIA Driver
- CUDA Toolkit
- cuDNN
- Python 3.8+
- Git

## Installation Steps

### 1. Update System
```bash
sudo yum update -y
sudo yum groupinstall "Development Tools" -y
```

### 2. Install NVIDIA Driver
```bash
# Add NVIDIA repository
sudo yum-config-manager --add-repo=https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo

# Install NVIDIA driver
sudo yum clean all
sudo yum install nvidia-driver-latest-dkms -y

# Verify installation
nvidia-smi
```

### 3. Install CUDA Toolkit
```bash
# Install CUDA repository
sudo yum-config-manager --add-repo=https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo

# Install CUDA
sudo yum clean all
sudo yum install cuda -y

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
nvcc --version
```

### 4. Install cuDNN
```bash
# Download cuDNN from NVIDIA website (requires NVIDIA account)
# Follow installation instructions from NVIDIA documentation
sudo yum install libcudnn8
```

### 5. Install Python and Dependencies
```bash
# Install Python 3.8+
sudo yum install python38 python38-devel -y

# Install pip
sudo yum install python38-pip -y

# Create virtual environment
python3.8 -m venv sam2_env
source sam2_env/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 6. Install Label Studio and ML Backend
```bash
# Install Label Studio
pip install label-studio

# Clone Label Studio ML backend
git clone https://github.com/humansignal/label-studio-ml-backend.git
cd label-studio-ml-backend
pip install -e .
```

### 7. Install SAM2
```bash
# Clone SAM2 repository
cd label-studio-ml-backend
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .

# Download model checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam2/sam2_h.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam2/sam2_l.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam2/sam2_b.pth
```

### 8. Configure Environment Variables
```bash
export LABEL_STUDIO_URL=http://localhost:8080
export LABEL_STUDIO_API_KEY=your_api_key_here
export DEVICE=cuda
```

### 9. Start Services

#### Start Label Studio
```bash
label-studio start
```

#### Start SAM2 ML Backend
```bash
cd label-studio-ml-backend/label_studio_ml/examples/segment_anything_2_image
label-studio-ml start .
```

## Verification and Testing

### 1. Verify Services
```bash
# Check Label Studio
curl http://localhost:8080

# Check ML Backend
curl http://localhost:9090
```

### 2. Connect SAM2 to Label Studio
1. Open Label Studio web interface
2. Go to Settings → Machine Learning
3. Add Model
4. Enter backend URL: `http://localhost:9090`
5. Click Validate and Save

## Troubleshooting

### Common Issues and Solutions

1. **NVIDIA Driver Issues**
   ```bash
   # Check NVIDIA driver status
   nvidia-smi
   # If failed, try reinstalling drivers
   sudo yum remove nvidia-driver-latest-dkms
   sudo yum install nvidia-driver-latest-dkms
   ```

2. **CUDA Version Mismatch**
   - Ensure PyTorch CUDA version matches installed CUDA version
   - Check CUDA version: `nvcc --version`
   - Check PyTorch CUDA: `python -c "import torch; print(torch.version.cuda)"`

3. **Memory Issues**
   - Monitor GPU memory: `nvidia-smi -l 1`
   - Adjust batch size if needed
   - Clear GPU cache if necessary

4. **Permission Issues**
   ```bash
   # Fix permission issues
   sudo chown -R $USER:$USER ~/.cache/torch
   sudo chown -R $USER:$USER ~/.cache/huggingface
   ```

## Maintenance

### Regular Updates
```bash
# Update system packages
sudo yum update -y

# Update Python packages
pip install --upgrade torch torchvision
pip install --upgrade label-studio label-studio-ml

# Update SAM2
cd segment-anything-2
git pull
pip install -e .
```

### Backup Configuration
- Regularly backup Label Studio database
- Save environment configurations
- Document any custom modifications

## Security Considerations

1. **Firewall Configuration**
   ```bash
   # Allow necessary ports
   sudo firewall-cmd --permanent --add-port=8080/tcp
   sudo firewall-cmd --permanent --add-port=9090/tcp
   sudo firewall-cmd --reload
   ```

2. **SSL/TLS Setup**
   - Configure SSL certificates for production use
   - Use HTTPS for all connections

3. **Access Control**
   - Use strong passwords
   - Implement IP whitelisting if needed
   - Regular security updates

## Performance Optimization

1. **GPU Optimization**
   - Monitor GPU utilization
   - Adjust batch sizes
   - Use appropriate model size (SAM2-H, SAM2-L, or SAM2-B)

2. **System Optimization**
   - Configure swap space
   - Optimize Python environment
   - Monitor system resources

## Additional Resources

- [SAM2 Official Documentation](https://github.com/facebookresearch/segment-anything-2)
- [Label Studio Documentation](https://labelstud.io/guide)
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda)
- [CentOS Documentation](https://docs.centos.org) 