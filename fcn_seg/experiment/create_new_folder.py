#useage: python create_new_folder.py OLDFILE NEWFILE

import sys,os
import shutil

if len(sys.argv) == 2:
	old_folder_name = os.path.join(sys.path[0],'exper_douyu1000_googlenet')
	new_folder_name = os.path.join(sys.path[0],sys.argv[1])
elif len(sys.argv) == 3:
	old_folder_name = os.path.join(sys.path[0],sys.argv[1])
	new_folder_name = os.path.join(sys.path[0],sys.argv[2])
print 'generate new folder \n[{}] \nfrom \n[{}]'.format(new_folder_name,old_folder_name)

if not os.path.exists(old_folder_name):
	print '[mkdir error]: {} does not exist!'.format(old_folder_name)
	sys.exit(0)

if not os.path.exists(new_folder_name):
	print '[mkdir]: {}'.format(new_folder_name)
	os.makedirs(new_folder_name)
	os.makedirs(os.path.join(new_folder_name,'snapshot'))
	old_file_list = os.listdir(old_folder_name)
	for file in old_file_list:
		if not file =='snapshot':
			shutil.copyfile(os.path.join(old_folder_name,file), \
				os.path.join(new_folder_name,file))
else:
	print '[mkdir failed]: {} has exist!'.format(new_folder_name)
	sys.exit(0)  

