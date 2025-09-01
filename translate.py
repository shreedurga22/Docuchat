import sys
from googletrans import Translator

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

translator = Translator()
print(translator.translate('Hello', dest='ta').text)
