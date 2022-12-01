import os
 
source = './FAM2A'
train_destination = './train_data'
test_destination = './test_data'
 
# gather all files
allfiles = os.listdir(source)

length = len(allfiles)
 
# iterate on all files to move them to destination folder
for i in range(0, int(length/75)):
    src_path = os.path.join(source, f)
    dst_path = os.path.join(train_destination, f)
    os.rename(src_path, dst_path)

for i in range(int(length/75), length):
    src_path = os.path.join(source, f)
    dst_path = os.path.join(test_destination, f)
    os.rename(src_path, dst_path)