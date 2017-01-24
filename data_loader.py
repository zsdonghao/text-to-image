
import os
import re
import time
import nltk
import tensorlayer as tl
from utils import *

dataset = '102flowers' #

if dataset == '102flowers':
    """
    images.shape = [8000, 64, 64, 3]
    captions_ids = [80000, any]
    """
    cwd = os.getcwd()
    img_dir = os.path.join(cwd, '102flowers/102flowers')
    caption_dir = os.path.join(cwd, '102flowers/text_c10')
    VOC_FIR = cwd + '/vocab.txt'

    ## load captions
    caption_sub_dir = load_folder_list( caption_dir )
    captions_dict = {}
    processed_capts = []
    for sub_dir in caption_sub_dir: # get caption file list
        with tl.ops.suppress_stdout():
            files = tl.files.load_file_list(path=sub_dir, regx='^image_[0-9]+\.txt')
            for i, f in enumerate(files):
                file_dir = os.path.join(sub_dir, f)
                key = int(re.findall('\d+', f)[0])
                t = open(file_dir,'r')
                lines = []
                for line in t:
                    lines.append(line.rstrip()) # remove \n
                    processed_capts.append(tl.nlp.process_sentence(line.rstrip(), start_word="<S>", end_word="</S>"))
                assert len(lines) == 10, "Every flower image have 10 captions"
                captions_dict[key] = lines
    print(" * %d x %d captions found " % (len(captions_dict), len(lines)))

    ## build vocab
    _ = tl.nlp.create_vocab(processed_capts, word_counts_output_file=VOC_FIR, min_word_count=1)
    vocab = tl.nlp.Vocabulary(VOC_FIR, start_word="<S>", end_word="</S>", unk_word="<UNK>")

    ## store all captions ids in list
    captions_ids = []
    try: # python3
        tmp = captions_dict.items()
    except: # python3
        tmp = captions_dict.iteritems()
    for key, value in tmp:
        for v in value:
            captions_ids.append( [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(v)] + [vocab.end_id])  # add END_ID
            # print(v)              # prominent purple stigma,petals are white inc olor
            # print(captions_ids)   # [[152, 19, 33, 15, 3, 8, 14, 719, 723]]
            # exit()
    captions_ids = np.asarray(captions_ids)
    print(" * tokenized %d captions" % len(captions_ids))

    ## check
    img_capt = captions_dict[1][1]
    print("img_capt: %s" % img_capt)
    print("nltk.tokenize.word_tokenize(img_capt): %s" % nltk.tokenize.word_tokenize(img_capt))
    img_capt_ids = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(img_capt)]#img_capt.split(' ')]
    print("img_capt_ids: %s" % img_capt_ids)
    print("id_to_word: %s" % [vocab.id_to_word(id) for id in img_capt_ids])

    ## load images
    with tl.ops.suppress_stdout():  # get image files list
        imgs_title_list = sorted(tl.files.load_file_list(path=img_dir, regx='^image_[0-9]+\.jpg'))
    print(" * %d images found, start loading and resizing ..." % len(imgs_title_list))
    s = time.time()

    # time.sleep(10)
    # def get_resize_image(name):   # fail
    #         img = scipy.misc.imread( os.path.join(img_dir, name) )
    #         img = tl.prepro.imresize(img, size=[64, 64])    # (64, 64, 3)
    #         img = img.astype(np.float32)
    #         return img
    # images = tl.prepro.threading_data(imgs_title_list, fn=get_resize_image)
    images = []
    for name in imgs_title_list:
        # print(name)
        img = scipy.misc.imread( os.path.join(img_dir, name) )
        img = tl.prepro.imresize(img, size=[64, 64])    # (64, 64, 3)
        img = img.astype(np.float32)
        images.append(img)
    images = np.asarray(images)
    print(" * loading and resizing took %ss" % (time.time()-s))

    n_images = len(captions_dict)
    n_captions = len(captions_ids)
    n_captions_per_image = len(lines) # 10

    print("n_captions: %d n_images: %d n_captions_per_image: %d" % (n_captions, n_images, n_captions_per_image))

    captions_ids_train, captions_ids_test = captions_ids[: 8000*n_captions_per_image], captions_ids[8000*n_captions_per_image :]
    images_train, images_test = images[:8000], images[8000:]
    n_images_train = len(images_train)
    n_images_test = len(images_test)
    n_captions_train = len(captions_ids_train)
    n_captions_test = len(captions_ids_test)
    print("n_images_train:%d n_captions_train:%d" % (n_images_train, n_captions_train))
    print("n_images_test:%d  n_captions_test:%d" % (n_images_test, n_captions_test))

    ## check test image
    # idexs = get_random_int(min=0, max=n_captions_test-1, number=64)
    # temp_test_capt = captions_ids_test[idexs]
    # for idx, ids in enumerate(temp_test_capt):
    #     print("%d %s" % (idx, [vocab.id_to_word(id) for id in ids]))
    # temp_test_img = images_train[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]
    # save_images(temp_test_img, [8, 8], 'temp_test_img.png')
    # exit()

    # ## check the first example
    # tl.visualize.frame(I=images[0], second=5, saveable=True, name='temp', cmap=None)
    # for cap in captions_dict[1]:
    #     print(cap)
    # print(captions_ids[0:10])
    # for ids in captions_ids[0:10]:
    #     print([vocab.id_to_word(id) for id in ids])
    # print_dict(captions_dict)

    # ## generate a random batch
    # batch_size = 64
    # idexs = get_random_int(0, n_captions_test, batch_size)
    # # idexs = [i for i in range(0,100)]
    # print(idexs)
    # b_seqs = captions_ids_test[idexs]
    # b_images = images_test[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]
    # print("before padding %s" % b_seqs)
    # b_seqs = tl.prepro.pad_sequences(b_seqs, padding='post')
    # print("after padding %s" % b_seqs)
    # # print(input_images.shape)   # (64, 64, 64, 3)
    # for ids in b_seqs:
    #     print([vocab.id_to_word(id) for id in ids])
    # print(np.max(b_images), np.min(b_images), b_images.shape)
    # from utils import *
    # save_images(b_images, [8, 8], 'temp2.png')
    # # tl.visualize.images2d(b_images, second=5, saveable=True, name='temp2')
    # exit()
