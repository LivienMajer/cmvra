from transformers import CLIPTokenizer
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import logging
from sklearn.metrics import balanced_accuracy_score
from zeta.train_classefier import MultiModalityClassifierTrainer
import numpy as np



def eval_text_encoder_process(visual_model, text_model, train_data, val_data, test_data, device, config):
    """
    Evaluate the text encoder process for a multi-modal model.

    This function tokenizes text descriptions, generates text embeddings,
    and either analyzes embeddings or computes accuracy on the dataset.

    Args:
        visual_model: The visual component of the multi-modal model.
        text_model: The text component of the multi-modal model.
        train_data, val_data, test_data: DataLoaders for respective datasets.
        device: The device to run computations on.
        config: Configuration dictionary containing settings.

   
    """
    
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

    if config['use_words']:
        try:
            text_descriptions = texts[config['word_key']]
        except KeyError:
            print(f"KeyError: '{config['word_key']}' not found in 'texts'. Available keys are: {list(texts.keys())}")
    else:
        text_descriptions = texts[config['dataset']]
    # Tokenize the text descriptions
    text_inputs = tokenizer(text_descriptions, return_tensors="pt", padding=True, truncation=True).to(device)

    # Create dummy pixel values
    batch_size = text_inputs['input_ids'].shape[0]


    # Obtain text embeddings using the text model of CLIP
    with torch.no_grad():
        text_outputs = text_model(input_ids=text_inputs['input_ids'])
        text_embeddings = text_outputs[1]
    if config.get('analyse_embbedings', False):
        trainer= MultiModalityClassifierTrainer(visual_model, device, train_data, val_data, test_data, config)
        trainer.analyze_embeddings(text_embeddings.cpu().numpy())
    else: 
        #get_accuracy(val_data, text_embeddings, visual_model, device, 'Val', config)
        #get_accuracy(train_data, text_embeddings, visual_model, device, 'Train', config)
        get_accuracy(test_data, text_embeddings, visual_model, device, 'Test', config)

    return 


@torch.no_grad()
def get_accuracy(dataset, text_embeddings, visual_model, device, info, config):
    def preprocess_for_clip_vip(data, modality):
        return data  # Assuming no special preprocessing is needed for CLIP-VIP

    def preprocess_for_mae(data, modality):
        return data.permute(0, 2, 1, 3, 4)  # Example permutation for MAE

    def preprocess_for_maeps(data, modality):
        return data.view(data.size(0), data.size(1), -1)

    def preprocess_for_omnivore(data, modality):
        if modality =='depth':
            return torch.cat((data, data[:, :, 0:1, :, :]), 2).permute(0, 2, 1, 3, 4)
        else:
            return data.permute(0, 2, 1, 3, 4)

    preprocessing_map = {
        'CLIP-VIP': preprocess_for_clip_vip,
        'MAE': preprocess_for_mae,
        'OMNIVORE': preprocess_for_omnivore,
        'DINO': preprocess_for_clip_vip,  # DINO requires the same preprocessing
        'MAEPS': preprocess_for_maeps
    }

    all_predictions = {modality: [] for modality in visual_model.module.modalities_encoders.keys()}
    all_labels = {modality: [] for modality in visual_model.module.modalities_encoders.keys()}

    correct_predictions_per_class = {modality: {} for modality in visual_model.module.modalities_encoders.keys()}
    visual_model.eval()
    text_embeddings = text_embeddings.cuda(device)

    for batch_data, labels in tqdm(dataset, desc=f"Evaluating"):
        #if 'rgb' in batch_data and 'rgb2' in batch_data:
        #    if not torch.equal(batch_data['rgb'], batch_data['rgb2']):
         #       print("Warning: Input data for 'modality1' and 'modality2' are not the same.")
        labels_cuda = labels.to(device)
        for modality, data in batch_data.items():
            if modality in visual_model.module.modalities_encoders:
                data = data.cuda(device, non_blocking=True)
                encoder = config['encoder_model'] if config['encoder_model'] != 'MIX' else config['modalities_encoders'].get(modality)
                if encoder in preprocessing_map:
                    data = preprocessing_map[encoder](data, modality)
                    outputs = visual_model.module.forward_encoder(modality, data)
                    predictions = torch.argmax(batch_cosine_similarity(outputs, text_embeddings), dim=1)
                    #print(modality)
                    #print(labels_cuda)
                    #print(predictions)
                    """

                    mask = (predictions == labels_cuda)
                    for label in labels_cuda.unique():
                        label_mask = (labels_cuda == label)
                        combined_mask = mask & label_mask
                        if label.item() not in correct_predictions_per_class[modality]:
                            correct_predictions_per_class[modality][label.item()] = combined_mask.sum()
                        else:
                            correct_predictions_per_class[modality][label.item()] += combined_mask.sum()
                    """

                    all_predictions[modality].append(predictions.cpu().numpy())
                    all_labels[modality].append(labels_cuda.cpu().numpy())

    # Print correct predictions per class
    """
    for modality in correct_predictions_per_class:
        logging.info(f"Modality: {modality}")
        for label in correct_predictions_per_class[modality]:
            logging.info(f"Class {label}: Correct Predictions - {correct_predictions_per_class[modality][label].sum().item()}")
    """
    avg_accuracies = {}
    avg_balanced_accuracies = {}
    for modality in all_predictions:
        if not all_predictions[modality]:
            logging.warning(f"No predictions for modality: {modality}. Skipping accuracy calculation.")
            avg_accuracies[modality] = None
            avg_balanced_accuracies[modality] = None
            continue
        modality_predictions = np.concatenate(all_predictions[modality])
        modality_labels = np.concatenate(all_labels[modality])
        modality_predictions_tensor = torch.tensor(modality_predictions, dtype=torch.long)
        modality_labels_tensor = torch.tensor(modality_labels, dtype=torch.long)

        avg_accuracies[modality] = compute_accuracy(modality_labels_tensor, modality_predictions_tensor)

        avg_balanced_accuracies[modality] = balanced_accuracy_score(modality_labels, modality_predictions)

    for modality in visual_model.module.modalities_encoders.keys():
        if avg_accuracies[modality] is not None and avg_balanced_accuracies[modality] is not None:
            logging.info(f"{info} Set: Modality: {modality}, Accuracy: {avg_accuracies[modality]:.4f}, Balanced Accuracy: {avg_balanced_accuracies[modality]:.4f}")
        else:
            logging.warning(f"Accuracy metrics not calculated for modality: {modality}.")

    return


def compute_accuracy(predicted_classes, true_labels):
    """
    Compute the accuracy of the predictions.

    Parameters:
    predicted_classes (torch.Tensor): Tensor containing the indices of the predicted classes.
    true_labels (torch.Tensor): Tensor containing the true labels.

    Returns:
    float: The accuracy of the predictions.
    """
    correct_predictions = (predicted_classes == true_labels).sum().item()
    #print(f"correct predictions:{correct_predictions}")
    total_predictions = true_labels.size(0)
    accuracy = correct_predictions / total_predictions
    #print(f"accuracy{accuracy}")
    return accuracy

def batch_cosine_similarity(x1, x2):
    # x1 has shape (batch_size, embed_dim)
    # x2 has shape (num_text_descriptions, embed_dim)
    dot = x1 @ x2.T
    norm1 = torch.norm(x1, p=2, dim=1).unsqueeze(1)
    norm2 = torch.norm(x2, p=2, dim=1).unsqueeze(0)
    return dot / (norm1 * norm2)



text_descriptions60 = [
"A person drinking water from a clear glass in a kitchen.",
"An individual eating a meal at a dining table, using a fork and knife.",
"A person brushing teeth with a toothbrush in a bathroom mirror.",
"Someone brushing long hair with a hairbrush in a bedroom.",
"A person dropping a red ball onto a wooden floor in a living room.",
"An individual picking up a blue book from the floor in a study room.",
"A person throwing a white paper airplane in an office setting.",
"Someone sitting down on a green armchair in a cozy room.",
"An individual standing up from a metal chair in a cafeteria.",
"A person clapping hands in an auditorium with a stage.",
"Someone reading a hardcover book in a library with bookshelves.",
"An individual writing in a notebook at a desk with a lamp.",
"A person tearing up a sheet of paper over a trash bin in a workspace.",
"Someone putting on a black jacket in a hallway with coat hangers.",
"An individual taking off a red jacket in a changing room.",
"A person putting on a white sneaker in a gym locker room.",
"Someone taking off a brown shoe in an entryway with a shoe rack.",
"An individual putting on eyeglasses in an office with a computer.",
"A person taking off sunglasses in a sunlit atrium.",
"Someone putting on a baseball cap in a sports store.",
"An individual taking off a wool hat in a coat room.",
"A person cheering up, smiling and laughing in a living room with a sofa.",
"Someone waving hand in a greeting at a hotel lobby.",
"An individual kicking a small football in an indoor play area.",
"A person reaching into a pocket of jeans in a bedroom.",
"Someone hopping on one foot in a fitness studio.",
"An individual jumping up with arms raised in a dance studio.",
"A person making a phone call on a smartphone in a home office.",
"Someone playing with a tablet on a couch in a family room.",
"An individual typing on a keyboard at a computer desk in a study.",
"A person pointing to a painting on a wall in an art gallery.",
"Someone taking a selfie with a phone in a mirror in a dressing room.",
"An individual checking time on a wristwatch in a conference room.",
"A person rubbing two hands together in a kitchen.",
"Someone nodding head in agreement in a meeting room with a whiteboard.",
"An individual shaking head in disapproval in a classroom.",
"A person wiping face with a handkerchief in a bathroom.",
"Someone saluting in a uniform in a military office.",
"An individual putting palms together in a gesture of prayer in a chapel.",
"A person crossing arms in front in a casual home setting.",
"Someone sneezing into a tissue in a doctor's waiting room.",
"An individual staggering in a hallway as if dizzy.",
"A person falling down onto a carpet in a living room.",
"Someone holding head in pain, indicating a headache, in an office.",
"An individual clutching chest in pain in a home living area.",
"A person holding lower back in pain in a furniture store.",
"Someone holding neck in pain in a home study.",
"An individual feeling nauseous, about to vomit, in a bathroom.",
"A person fanning self with a magazine in a warm room.",
"Someone punching the air in a boxing gym.",
"An individual kicking a pillow in a bedroom.",
"A person pushing a chair in a dining room.",
"Someone patting a friend on the back in a coffee shop.",
"An individual pointing a finger at a computer screen in an office.",
"A person hugging a friend in a living room.",
"Someone giving a pen to another person in an office.",
"An individual touching the pocket of their jeans in a bedroom.",
"Two people shaking hands in a business meeting room.",
"A person walking towards a window in a bright room.",
"Two individuals walking apart in a hallway of an office building."
]


text_descriptions120 = [
    "A person drinking water from a clear glass in a kitchen.",
    "An individual eating a meal at a dining table, using a fork and knife.",
    "A person brushing teeth with a toothbrush in a bathroom mirror.",
    "Someone brushing long hair with a hairbrush in a bedroom.",
    "A person dropping a red ball onto a wooden floor in a living room.",
    "An individual picking up a blue book from the floor in a study room.",
    "A person throwing a white paper airplane in an office setting.",
    "Someone sitting down on a green armchair in a cozy room.",
    "An individual standing up from a metal chair in a cafeteria.",
    "A person clapping hands in an auditorium with a stage.",
    "Someone reading a hardcover book in a library with bookshelves.",
    "An individual writing in a notebook at a desk with a lamp.",
    "A person tearing up a sheet of paper over a trash bin in a workspace.",
    "Someone putting on a black jacket in a hallway with coat hangers.",
    "An individual taking off a red jacket in a changing room.",
    "A person putting on a white sneaker in a gym locker room.",
    "Someone taking off a brown shoe in an entryway with a shoe rack.",
    "An individual putting on eyeglasses in an office with a computer.",
    "A person taking off sunglasses in a sunlit atrium.",
    "Someone putting on a baseball cap in a sports store.",
    "An individual taking off a wool hat in a coat room.",
    "A person cheering up, smiling and laughing in a living room with a sofa.",
    "Someone waving hand in a greeting at a hotel lobby.",
    "An individual kicking a small football in an indoor play area.",
    "A person reaching into a pocket of jeans in a bedroom.",
    "Someone hopping on one foot in a fitness studio.",
    "An individual jumping up with arms raised in a dance studio.",
    "A person making a phone call on a smartphone in a home office.",
    "Someone playing with a tablet on a couch in a family room.",
    "An individual typing on a keyboard at a computer desk in a study.",
    "A person pointing to a painting on a wall in an art gallery.",
    "Someone taking a selfie with a phone in a mirror in a dressing room.",
    "An individual checking time on a wristwatch in a conference room.",
    "A person rubbing two hands together in a kitchen.",
    "Someone nodding head in agreement in a meeting room with a whiteboard.",
    "An individual shaking head in disapproval in a classroom.",
    "A person wiping face with a handkerchief in a bathroom.",
    "Someone saluting in a uniform in a military office.",
    "An individual putting palms together in a gesture of prayer in a chapel.",
    "A person crossing arms in front in a casual home setting.",
    "Someone sneezing into a tissue in a doctor's waiting room.",
    "An individual staggering in a hallway as if dizzy.",
    "A person falling down onto a carpet in a living room.",
    "Someone holding head in pain, indicating a headache, in an office.",
    "An individual clutching chest in pain in a home living area.",
    "A person holding lower back in pain in a furniture store.",
    "Someone holding neck in pain in a home study.",
    "An individual feeling nauseous, about to vomit, in a bathroom.",
    "A person fanning self with a magazine in a warm room.",
    "Someone punching the air in a boxing gym.",
    "An individual kicking a pillow in a bedroom.",
    "A person pushing a chair in a dining room.",
    "Someone patting a friend on the back in a coffee shop.",
    "An individual pointing a finger at a computer screen in an office.",
    "A person hugging a friend in a living room.",
    "Someone giving a pen to another person in an office.",
    "An individual touching the pocket of their jeans in a bedroom.",
    "Two people shaking hands in a business meeting room.",
    "A person walking towards a window in a bright room.",
    "Two individuals walking apart in a hallway of an office building.",
    "A person putting on headphones in a quiet study room",
    "An individual taking off headphones in a home office with a computer.",
    "Someone shooting a basketball towards a hoop in an outdoor court during sunset.",
    "A person bouncing a ball on a paved driveway with a basketball hoop in the background.",
    "An individual swinging a tennis racket at a yellow ball on a sunny tennis court.",
    "Someone juggling table tennis balls in a bright game room with a ping pong table.",
    "A person gesturing 'hush' with a finger on lips in a library filled with bookshelves.",
    "An individual flicking their long hair back in a mirror reflection in a dance studio.",
    "A person giving a thumbs up in a bright classroom with students and a chalkboard.",
    "Someone giving a thumbs down in a meeting room with a large monitor displaying data.",
    "An individual making an OK sign with their hand in a cozy cafe with coffee cups on tables.",
    "A person making a victory sign with their fingers in front of a scenic viewpoint overlooking mountains.",
    "Someone stapling pages of a book in a crafting room with art supplies on shelves.",
    "An individual counting money on a wooden table in a small business office.",
    "A person cutting nails sitting on a porch with a garden view.",
    "Someone cutting paper with scissors on a cluttered craft table in an art studio.",
    "An individual snapping fingers to music in a brightly lit kitchen while cooking.",
    "A person opening a bottle in a picnic setting with a basket and a blanket.",
    "Someone sniffing a perfume bottle in a boutique with shelves of beauty products.",
    "An individual squatting down to tie shoelaces in a gym with workout equipment.",
    "A person tossing a coin into a fountain outdoors with trees in the background.",
    "Someone folding paper into an airplane in a classroom with kids' drawings on the walls.",
    "An individual balling up paper in frustration in an office with a computer and documents.",
    "A person playing with a magic cube in a cozy living room on a soft rug.",
    "Someone applying cream on their face in a bright bathroom with a large mirror.",
    "An individual applying cream on the back of their hand in a beauty salon with products on display.",
    "A person putting on a backpack in a hostel dormitory with bunk beds and lockers.",
    "Someone taking off a backpack at the entrance of a hiking trail with woods in the background.",
    "An individual putting something into a bag on a kitchen counter with groceries around.",
    "A person taking something out of a bag in a classroom with desks and a projector.",
    "Someone opening a box in a living room on Christmas morning with decorations.",
    "An individual moving heavy objects in a garage filled with tools and storage boxes.",
    "A person shaking their fist in excitement at a sports event with a crowd cheering.",
    "Someone throwing up a cap in celebration during a graduation ceremony in an open field.",
    "An individual with hands up in surrender during a playful water fight in a backyard.",
    "A person crossing their arms while waiting in a coffee shop with a line of customers.",
    "Someone doing arm circles as part of a warm-up in a fitness class at a gym.",
    "An individual performing arm swings in a park with autumn leaves on the ground.",
    "A person running on the spot in a home gym with a treadmill and weights.",
    "Someone doing butt kicks during a track workout on a sunny day at an athletic field.",
    "An individual performing a cross toe touch in a yoga studio with mats and calming decor.",
    "A person executing a side kick in a martial arts dojo with mirrors and training pads.",
    "Someone yawning widely in a cozy bedroom early in the morning with the sun rising.",
    "An individual stretching themselves in an office during a break with a coffee cup on the desk.",
    "A person blowing their nose with a tissue in a bright, airy living room with a large window.",
    "Someone hitting another person with a foam bat in a playful outdoor party setting.",
    "An individual wielding a knife towards another person in a dramatic theater rehearsal scene.",
    "A person knocking over another person during a friendly beach volleyball game.",
    "Someone grabbing another person’s hat in a playful manner at a sunny park.",
    "An individual shooting at another person with a water gun during a summer backyard party.",
    "A person stepping on someone’s foot accidentally in a crowded subway car.",
    "Someone high-fiving in a sports team huddle on a field with goals in the background.",
    "An individual cheering and drinking with friends at a rooftop bar with city lights.",
    "Two people carrying a couch together into a new apartment with boxes around.",
    "A person taking a photo of another person in front of a famous landmark during a trip.",
    "Someone following another person in a vast open room.",
    "An individual whispering in another person’s ear during a secret exchange in a library.",
    "Two people exchanging things in an industrial building.",
    "Someone supporting somebody with a hand during a difficult hiking trail with scenic views.",
    "Two individuals playing rock-paper-scissors in a schoolyard with children watching."
]


daa_descriptions = ['Closing the car door from outside the vehicle',
 'Opening the car door from outside the vehicle',
 'Entering the vehicle and getting seated',
 'Closing the car door from inside the vehicle',
 'Fastening the seat belt before starting the journey',
 "Interacting with the car's multimedia display",
 'Sitting still and not engaging in any other activity',
 'Pressing a button for automated vehicle features',
 'Reaching for and fetching an object inside the car',
 'Opening a laptop to begin work or entertainment',
 'Typing and focusing on tasks on the laptop',
 'Using a smartphone for calling, texting, or other apps',
 'Consuming a beverage, perhaps from a travel mug or bottle',
 'Shutting the laptop after finishing tasks',
 'Placing an object down in a designated spot within the car',
 'Unfastening the seat belt, possibly in preparation to exit',
 'Putting on a jacket or coat while seated in the car',
 'Opening a bottle, perhaps to drink or serve to others',
 'Sealing a bottle after use',
 'Looking around the interior or shifting position',
 'Preparing or arranging food, possibly for a meal on the go',
 'Consuming food while seated inside the vehicle',
 'Removing sunglasses, possibly as light conditions change',
 'Putting on sunglasses for comfort or style',
 'Reading a newspaper for news or entertainment',
 'Writing notes or filling out documents',
 'Engaging in a conversation using a mobile phone',
 'Reading a magazine for leisure or information',
 'Removing a jacket or coat, perhaps due to changing temperatures',
 'Opening the car door from the inside to exit or for fresh air',
 'Exiting the vehicle upon reaching the destination',
 'Opening a backpack to retrieve or store items',
 'Storing the laptop in a backpack after use',
 'Retrieving the laptop from a backpack to use']

text_descriptions60words = [
    "drink water",
    "eat meal",
    "brush teeth",
    "brush hair",
    "drop",
    "pick up",
    "throw",
    "sit down",
    "stand up",
    "clapping",
    "reading",
    "writing",
    "tear up paper",
    "put on jacket",
    "take off jacket",
    "put on a shoe",
    "take off a shoe",
    "put on glasses",
    "take off glasses",
    "put on a hat/cap",
    "take off a hat/cap",
    "cheer up",
    "hand waving",
    "kicking something",
    "reach into pocket",
    "hopping",
    "jump up",
    "phone call",
    "play with phone/tablet",
    "typing on a keyboard",
    "point to something",
    "taking a selfie",
    "check time (from watch)",
    "rub two hands",
    "nod head/bow",
    "shake head",
    "wipe face",
    "salute",
    "put palms together",
    "cross hands in front",
    "sneeze/cough",
    "staggering",
    "falling down",
    "headache",
    "chest pain",
    "back pain",
    "neck pain",
    "nausea/vomiting",
    "fan self",
    "punch/slapp",
    "kicking",
    "pushing",
    "pat on back",
    "point finger",
    "hugging",
    "giving object",
    "touch pocket",
    "shaking hands",
    "walking towards",
    "walking apart"
]

text_descriptions60_120words = [
    "put on headphone",
    "take off headphone",
    "shoot at the basket",
    "bounce ball",
    "tennis bat swing",
    "juggling table tennis balls",
    "hush (quite)",
    "flick hair",
    "thumb up",
    "thumb down",
    "make ok sign",
    "make victory sign",
    "staple book",
    "counting money",
    "cutting nails",
    "cutting paper (using scissors)",
    "snapping fingers",
    "open bottle",
    "sniff (smell)",
    "squat down",
    "toss a coin",
    "fold paper",
    "ball up paper",
    "play magic cube",
    "apply cream on face",
    "apply cream on hand back",
    "put on bag",
    "take off bag",
    "put something into a bag",
    "take something out of a bag",
    "open a box",
    "move heavy objects",
    "shake fist",
    "throw up cap/hat",
    "hands up (both hands)",
    "cross arms",
    "arm circles",
    "arm swings",
    "running on the spot",
    "butt kicks (kick backward)",
    "cross toe touch",
    "side kick",
    "yawn",
    "stretch oneself",
    "blow nose",
    "hit other person with something",
    "wield knife towards other person",
    "knock over other person (hit with body)",
    "grab other person’s stuff",
    "shoot at other person with a gun",
    "step on foot",
    "high-five",
    "cheers and drink",
    "carry something with other person",
    "take a photo of other person",
    "follow other person",
    "whisper in other person’s ear",
    "exchange things with other person",
    "support somebody with hand",
    "finger-guessing game (playing rock-paper-scissors)"
]

daa_descriptions_words = [
'closing_door_outside',
 'opening_door_outside',
 'entering_car',
 'closing_door_inside',
 'fastening_seat_belt',
 'using_multimedia_display',
 'sitting_still',
 'pressing_automation_button',
 'fetching_an_object',
 'opening_laptop',
 'working_on_laptop',
 'interacting_with_phone',
 'drinking',
 'closing_laptop',
 'placing_an_object',
 'unfastening_seat_belt',
 'putting_on_jacket',
 'opening_bottle',
 'closing_bottle',
 'looking_or_moving_around',
 'preparing_food',
 'eating',
 'taking_off_sunglasses',
 'putting_on_sunglasses',
 'reading_newspaper',
 'writing',
 'talking_on_phone',
 'reading_magazine',
 'taking_off_jacket',
 'opening_door_inside',
 'exiting_car',
 'opening_backpack',
 'putting_laptop_into_backpack',
 'taking_laptop_from_backpack']

zs0 = ['taking_off_sunglasses', 'eating', 'talking_on_phone', 'writing', 'working_on_laptop', 'preparing_food', 'taking_off_jacket', 'opening_bottle', 'placing_an_object', 'closing_laptop']

zs1 = ['eating', 'fastening_seat_belt', 'opening_backpack', 'taking_off_sunglasses', 'entering_car', 'pressing_automation_button', 'opening_door_outside', 'opening_laptop', 'drinking', 'putting_on_jacket']

zs2 = ['putting_on_jacket', 'sitting_still', 'using_multimedia_display', 'closing_laptop', 'closing_door_inside', 'looking_or_moving_around (e.g. searching)', 'opening_door_outside', 'opening_laptop', 'eating', 'taking_off_jacket']

zs3 = ['preparing_food', 'closing_bottle', 'entering_car', 'putting_on_jacket', 'opening_door_inside', 'reading_magazine', 'closing_laptop', 'taking_off_sunglasses', 'opening_bottle', 'fastening_seat_belt']

zs4 = ['preparing_food', 'closing_door_outside', 'exiting_car', 'putting_on_sunglasses', 'entering_car', 'fastening_seat_belt', 'opening_door_outside', 'fetching_an_object', 'using_multimedia_display', 'talking_on_phone']

zs5 = ['unfastening_seat_belt', 'fetching_an_object', 'writing', 'reading_magazine', 'exiting_car', 'fastening_seat_belt', 'working_on_laptop', 'entering_car', 'preparing_food', 'opening_door_outside']

zs6 = ['opening_door_outside', 'reading_magazine', 'reading_newspaper', 'closing_bottle', 'putting_on_jacket', 'eating', 'closing_laptop', 'opening_bottle', 'preparing_food', 'fetching_an_object']

zs7 = ['drinking', 'taking_off_sunglasses', 'talking_on_phone', 'putting_on_jacket', 'fetching_an_object', 'closing_door_outside', 'opening_bottle', 'opening_door_inside', 'unfastening_seat_belt', 'putting_on_sunglasses']

zs8 = ['taking_off_jacket', 'closing_door_outside', 'drinking', 'looking_or_moving_around (e.g. searching)', 'reading_magazine', 'exiting_car', 'reading_newspaper', 'opening_bottle', 'closing_door_inside', 'opening_door_inside']

zs9 = ['working_on_laptop', 'closing_door_outside', 'interacting_with_phone', 'reading_newspaper', 'closing_laptop', 'looking_or_moving_around (e.g. searching)', 'unfastening_seat_belt', 'sitting_still', 'talking_on_phone', 'putting_on_jacket']


texts = {
    'NTU120Words': text_descriptions60words + text_descriptions60_120words,
    'NTU60-120Words': text_descriptions60_120words,
    'NTUWords':text_descriptions60words,
     'NTU': text_descriptions60,
     'NTU120': text_descriptions120,
     'DAA': daa_descriptions,
     'DAAWords': daa_descriptions_words,
     'DAA_ZS0': zs0,
    'DAA_ZS1': zs1,
    'DAA_ZS2': zs2,
    'DAA_ZS3': zs3,
    'DAA_ZS4': zs4,
    'DAA_ZS5': zs5,
    'DAA_ZS6': zs6,
    'DAA_ZS7': zs7,
    'DAA_ZS8': zs8,
    'DAA_ZS9': zs9
}