import logging

from src.utils.data_functions import *

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)



def validate():
    """Validate the model on the validation data set and return the score.
    
    Returns:
        score (float): The score of the model on the validation data set
    """
    validation_report = 'validation_report.md'

    logger.info('Loading data')
    X, y = load_data('data/validation.csv')

    logger.info('Loading model')
    model = load_model('models/best_model.pkl')

    logger.info('validating model')
    score = model.score(X, y)

    logger.info('The model has a score of %s on validation data', score)

    generate_validation_report(model, X, y)

    logger.info('Validation report generated in %s', validation_report)


    return score


if __name__ == '__main__':
    validate()