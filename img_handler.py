from PIL import Image

class ImageHandler:
    _instance = None
    _target_height = 500  # Adjust this as needed

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.load_images()
        return cls._instance

    def load_images(self):
        self._yes_img = self.load_and_resize_image('resources/yes.png')
        self._no_img = self.load_and_resize_image('resources/no.png')

    def load_and_resize_image(self, image_path):
        image = Image.open(image_path)
        # new_width = int(image.width * self._target_height / image.height)
        # image = image.resize((new_width, self._target_height))
        return image

    @staticmethod
    def get_image(input_type):
        instance = ImageHandler()  # Ensure initialization
        if input_type == 'yes':
            return instance._yes_img
        else:
            return instance._no_img
