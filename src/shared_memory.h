#ifndef AOD_SHARED_MEMORY_H
#define AOD_SHARED_MEMORY_H

#include <string>
#include <cstddef>

namespace aod {

// POSIX shared memory manager for zero-copy client communication
// Manages lifecycle of shared memory regions for array data transfer
class SharedMemoryManager {
public:
    SharedMemoryManager();
    ~SharedMemoryManager();

    // Create and map shared memory region
    // name: POSIX shm name (e.g., "/aod_server_12345")
    // size: Size in bytes
    // Returns true on success
    bool create(const std::string& name, size_t size);

    // Unmap and unlink shared memory
    void destroy();

    // Get pointer to mapped region
    void* getPointer() const { return ptr_; }

    // Get region name
    std::string getName() const { return name_; }

    // Get size
    size_t getSize() const { return size_; }

    // Check if created
    bool isCreated() const { return ptr_ != nullptr; }

private:
    std::string name_;  // Shared memory object name
    void* ptr_;         // Mapped memory pointer
    size_t size_;       // Size in bytes
    int fd_;            // File descriptor from shm_open
};

} // namespace aod

#endif // AOD_SHARED_MEMORY_H
