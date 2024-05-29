from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Modeli yükle
model = load_model('tomato_ripeness_model.h5')

# Örnek bir görüntü yükle
img_path = './old.jpg'
img = image.load_img(img_path, target_size=(256, 256))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Sınıf tahmini yap
classes = model.predict(x)
print(classes)

#İlk eleman: Ripe (Olgun) tomates olasılığı
#İkinci eleman: Unripe (Olgunlaşmamış) tomates olasılığı
#Üçüncü eleman: Old (Eskimiş) tomates olasılığı
#Dördüncü eleman: Damaged (Hasarlı) tomates olasılığı