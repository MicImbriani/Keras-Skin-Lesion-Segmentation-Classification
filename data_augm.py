import numpy as np

np.random.seed(42)
%%time
# DATA_DIR = 'D:/Users/imbrm/ISIC_2017-2'
train = pd.read_h(DATA_DIR + 'tgs_data.h5', key='filtered_train')
train.head()

NUM_CROPS = 25    # Number of crops for every image
WIDTH = 64
HEIGHT = 64
IMAGE_PIXELS = WIDTH * HEIGHT
MAX_X = 37
MAX_Y = 37

def coverage(pixels):
    if pixels == 0:
        return 0
    else:
        percentage = pixels / IMAGE_PIXELS
        return np.ceil(percentage * 10).astype(np.uint)

def random_crop_params():
    x = np.random.randint(0, MAX_X)
    y = np.random.randint(0, MAX_Y)
    flip = np.random.choice(a=[False, True])
    intensity = np.random.normal(1,0.2)
    if intensity == 0:
        intensity = 0
    return x, y, flip, intensity

def crop(img, x, y, flip, intensity=1):
    random_img = img[x:x+WIDTH,y:y+WIDTH]
    if flip:
        random_img = np.fliplr(random_img)
    random_img = random_img * intensity
    return random_img.reshape(WIDTH, HEIGHT, 1)

imgs_aug = []
masks_aug = []
for idx in train.index:
    img = train.loc[idx]['images']
    mask = train.loc[idx]['masks']
    for i in range(NUM_CROPS):
        r = i // 5
        c = i % 5
        x, y, flip, intensity = random_crop_params()
        random_img = crop(img, x, y, flip, intensity)
        imgs_aug.append(random_img)
        random_mask = crop(mask, x, y, flip)
        masks_aug.append(random_mask)

data = pd.DataFrame({'images':imgs_aug, 'masks':masks_aug})
print(data.shape)

# Calculate coverage
data['pixels'] = data['masks'].map(lambda x: np.sum(x/255)).astype(np.int16)
data['coverage'] = data['pixels'].map(coverage).astype(np.float16)

data.describe()

del imgs_aug
del masks_aug
del train


# Plot coverage distribution
labels, counts = np.unique(data['coverage'], return_counts=True)
plt.bar(labels, counts, align='center')
plt.gca().set_xticks(labels)
plt.grid(axis='y')
plt.show()