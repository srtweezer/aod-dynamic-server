#include "shared_memory.h"
#include <iostream>
#include <cstring>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace aod {

SharedMemoryManager::SharedMemoryManager()
    : ptr_(nullptr),
      size_(0),
      fd_(-1) {
}

SharedMemoryManager::~SharedMemoryManager() {
    destroy();
}

bool SharedMemoryManager::create(const std::string& name, size_t size) {
    name_ = name;
    size_ = size;

    std::cout << "[Shared Memory] Creating shared memory: " << name << std::endl;
    std::cout << "[Shared Memory]   Size: " << size / (1024*1024) << " MB" << std::endl;

    // Create shared memory object
    fd_ = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd_ == -1) {
        std::cerr << "[Shared Memory] shm_open failed: " << strerror(errno) << std::endl;
        return false;
    }

    // Set size
    if (ftruncate(fd_, size) == -1) {
        std::cerr << "[Shared Memory] ftruncate failed: " << strerror(errno) << std::endl;
        close(fd_);
        shm_unlink(name.c_str());
        fd_ = -1;
        return false;
    }

    // Map into memory
    ptr_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (ptr_ == MAP_FAILED) {
        std::cerr << "[Shared Memory] mmap failed: " << strerror(errno) << std::endl;
        close(fd_);
        shm_unlink(name.c_str());
        fd_ = -1;
        ptr_ = nullptr;
        return false;
    }

    // Zero the memory initially
    std::memset(ptr_, 0, size);

    std::cout << "[Shared Memory] Created successfully at " << ptr_ << std::endl;
    return true;
}

void SharedMemoryManager::destroy() {
    if (ptr_ != nullptr && ptr_ != MAP_FAILED) {
        std::cout << "[Shared Memory] Unmapping memory..." << std::endl;
        munmap(ptr_, size_);
        ptr_ = nullptr;
    }

    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }

    if (!name_.empty()) {
        std::cout << "[Shared Memory] Unlinking " << name_ << std::endl;
        shm_unlink(name_.c_str());
        name_ = "";
    }

    size_ = 0;
}

} // namespace aod
