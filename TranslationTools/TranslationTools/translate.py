import sys
from googletrans import Translator

# Configure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# Translate and print
translator = Translator()
translation = translator.translate("Hello", dest='ta').text
print(translation)
