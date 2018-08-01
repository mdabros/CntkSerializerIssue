# CntkSerializerIssue

Repository for reproducing issue: [3280: "Bug: MinibatchSource.GetNextMinibatch hangs (multiple serializers?"](https://github.com/Microsoft/CNTK/issues/3280)

### Running the code
1. Clone this repository.
2. Open the solution in visual studio 2017.
3. Build and run the project **CntkSerializerIssue**

### Images
Image data is located in the **images** folder in the root of the repository. 
There are 4 directories, one for each channel:
 - Channel1
 - Channel2
 - Channel3
 - Channel4

Each directory contains the first 128 images of the "zero" class from the mnist data set. 
The original data set can be found here: [http://yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/)
So each channel contains the same set of gray level images. 
This does not matter for this case, since we are only trying to reproduce minibatch source/serializer issue.

### Map and ctf files
The map and ctf files for training are located in the **mapfiles** folder in the root of the repository.