#include <iostream>
using namespace std;

struct blah {
  int a;
};
void createStruct(blah ***blah_array) {

//blah ** createStruct() {

  //blah ** blah_array;
  blah ** temp;
  temp = new blah* [10];
  *blah_array = temp;
  cout<<"blah array in function "<<*blah_array<<endl;
  for (int i=0;i<10;i++) {
    (*blah_array)[i] = new blah;
    cout<<"created "<<i<<endl;
    (*blah_array)[i]->a = 5;
  }
  //return blah_array;
}
int main(int argc, char** argv){
  blah** blah_array;
  cout<<"before the function blah array is "<<blah_array<<endl;
  //blah_array = new blah*[10];
  createStruct(&blah_array);
  cout<<"blah array is "<<blah_array<<endl;
  //blah_array = createStruct();
  for (int i=0;i<10;i++){
    cout<<"printing"<<endl;
    cout<<i<<":"<<blah_array[i]->a<<endl;
    //cout<<i<<":"<<(*(blah_array + sizeof(blah*)*i))->a<<endl;
  }
}
