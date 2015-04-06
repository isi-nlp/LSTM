#ifdef _OPENMP
# include <omp.h>
struct MutexType
{
    MutexType() { omp_init_lock(&lock); }
    ~MutexType() { omp_destroy_lock(&lock); }
    void Lock() { omp_set_lock(&lock); }
    void Unlock() { omp_unset_lock(&lock); }
   
    MutexType(const MutexType& ) { omp_init_lock(&lock); }
    MutexType& operator= (const MutexType& ) { return *this; }
    public:
    omp_lock_t lock;
};
#else
/* A dummy mutex that doesn't actually exclude anything,
* but as there is no parallelism either, no worries. */
struct MutexType
{
    void Lock() {}
    void Unlock() {}
};
#endif

/* An exception-safe scoped lock-keeper. */
struct ScopedLock
{
    explicit ScopedLock(MutexType& m) : mut(m), locked(true) { mut.Lock(); }
    ~ScopedLock() { Unlock(); }
    void Unlock() { if(!locked) return; locked=false; mut.Unlock(); }
    void LockAgain() { if(locked) return; mut.Lock(); locked=true; }
    private:
        MutexType& mut;
        bool locked;
   private: // prevent copying the scoped lock.
        void operator=(const ScopedLock&);
        ScopedLock(const ScopedLock&);
};

