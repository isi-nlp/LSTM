#include <set>

class data
{
private:
 std::set<int> flags;
 MutexType lock;
public:
 bool set_get(int c)
 {
   ScopedLock lck(lock); // locks the mutex
   
   if(flags.find(c) != flags.end()) return true; // was found
   flags.insert(c);
   return false; // was not found
 } // automatically releases the lock when lck goes out of scope.
};

