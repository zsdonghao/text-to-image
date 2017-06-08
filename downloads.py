from __future__ import print_function
import os,sys,gzip,requests,zipfile,tarfile
from tqdm import tqdm
from six.moves import urllib
import time

'''
This script is mainly used in cooperation with codes from https://github.com/zsdonghao/text-to-image
download flower dataset from : http://www.robots.ox.ac.uk/~vgg/data/flowers/102/
download caption dataset from : https://drive.google.com/uc?export=download&confirm=l7Ld&id=0B0ywwgffWnLLcms2WWJQRFNSWXM
'''


def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value
	return None

def save_response_content(response, destination, chunk_size=32*1024):
	total_size = int(response.headers.get('content-length', 0))
	with open(destination, "wb") as f:
		for chunk in tqdm(response.iter_content(chunk_size), total=total_size,
				unit='B', unit_scale=True, desc=destination):
			if chunk: # filter out keep-alive new chunks
				f.write(chunk)

def download_file_from_google_drive(id, destination):    
	URL = "https://docs.google.com/uc?export=download"
	session = requests.Session()

	response = session.get(URL, params={ 'id': id }, stream=True)
	token = get_confirm_token(response)

	if token:
		params = { 'id' : id, 'confirm' : token }
		response = session.get(URL, params=params, stream=True)
	save_response_content(response, destination)

def download_caption(dirpath):
	data_dir = 'cvpr2016_flowers.tar.gz'
	if os.path.exists(os.path.join(dirpath, data_dir)):
		print('Found cvpr2016_flowers.tar.gz - skip')
		return

	filename, drive_id  = "cvpr2016_flowers.tar.gz", "0B0ywwgffWnLLcms2WWJQRFNSWXM"
	save_path = os.path.join(dirpath, filename)

	if os.path.exists(save_path):
		print('[*] {} already exists'.format(save_path))
	else:
		download_file_from_google_drive(drive_id, save_path)


def download(url, dirpath):
	filepath = dirpath
	u = urllib.request.urlopen(url)
	f = open(filepath, 'wb')
	filesize = int(u.headers["Content-Length"])
	print("Downloading: %s Bytes: %s" % ("102flowers", filesize))

	downloaded = 0
	block_sz = 8192
	status_width = 70
	while True:
		buf = u.read(block_sz)
		if not buf:
			print('')
			break
		else:
			print('', end='\r')
		downloaded += len(buf)
		f.write(buf)

		status = (("[{}  " + " ***progress: {:03.1f}% ]").format('=' * int(float(downloaded) / 
			filesize * status_width) + '>', downloaded * 100. / filesize))
		print(status, end='')

		sys.stdout.flush()
	f.close()
	return filepath

def unzip(src_dir,new_name = None):
	# extract to current directory
	dirpath = '.'
	try:
		if src_dir.endswith('.zip'):
			print('unzipping ' + src_dir)
			with zipfile.ZipFile(src_dir) as zf:
				zip_dir = zf.namelist()[0]
				zf.extractall(dirpath)
		elif src_dir.endswith('.tgz') or src_dir.endswith('tar.gz'):
			print('unzipping ' + src_dir)
			tar = tarfile.open(src_dir)
			tar.extractall()
			tar.close()
		# os.remove(save_path)
		if new_name is None:
			pass
		else:
			os.rename('jpg', os.path.join(dirpath, new_name))
	except:
		raise('wrong format')

def main():
	url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
	cur_dir = os.getcwd()
	image_dir = os.path.join(cur_dir,"102flowers.tgz")
	if os.path.exists(image_dir):
		print('dataset already exists')
	else:
		download(url,image_dir)
	unzip(image_dir,'102flowers')

	caption_dir = os.path.join(cur_dir,"cvpr2016_flowers.tar.gz")
	if os.path.exists(caption_dir):
		print('dataset already exists')
	else:
		download_caption(cur_dir)
	unzip(caption_dir)

if __name__ == '__main__':
	main()
	
	