from utils.celery_app import celery 
from utils.init_model import init_model_sync
import utils.tasks

init_model_sync()

app = celery