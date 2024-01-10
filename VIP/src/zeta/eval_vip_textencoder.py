from transformers import CLIPTokenizer
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import logging

def eval_text_encoder_process(visual_model, text_model, train_data, val_data, test_data, device, config):
    
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

    # Tokenize the text descriptions
    text_inputs = tokenizer(text_descriptions, return_tensors="pt", padding=True, truncation=True).to(device)

    # Create dummy pixel values
    batch_size = text_inputs['input_ids'].shape[0]


    # Obtain text embeddings using the text model of CLIP
    with torch.no_grad():
        text_outputs = text_model(input_ids=text_inputs['input_ids'])
        text_embeddings = text_outputs[1]

    get_accuarcy(val_data, text_embeddings, visual_model, device, 'Val')
    get_accuarcy(train_data, text_embeddings, visual_model, device, 'Train')
    get_accuarcy(test_data, text_embeddings, visual_model, device, 'Test')

    return 


@torch.no_grad()
def get_accuarcy(dataset, text_embeddings, visual_model, device, info):
    epoch_accuracies = {modality: 0.0 for modality in visual_model.module.modalities_encoders.keys()}

    visual_model.eval()
    for batch_data, batch_labels in tqdm(dataset, desc=f"Evaluing"):
        for modality in batch_data:
            if modality in visual_model.module.modalities_encoders:
                data = batch_data[modality].cuda(device)
                labels = batch_labels.cuda(device)

                outputs = visual_model.module.forward_encoder(modality, data)
                predictions = torch.argmax(batch_cosine_similarity(outputs, text_embeddings), dim=1)
                #print(predictions)
                #print(labels)
                accuracy = compute_accuracy(predictions, labels)
                
                epoch_accuracies[modality] += accuracy
                #print(f"epoch_accuracies[modality] {epoch_accuracies[modality]}")
                #print(len(dataset))
    # Calculate average accuracy for each modality
    
    avg_accuracies = {modality: epoch_accuracies[modality] / len(dataset) for modality in epoch_accuracies}

    
    for modality in visual_model.module.modalities_encoders:
        logging.info(f"{info} Set: Modality: {modality}, Accuracy: {avg_accuracies[modality]:.4f}")

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



text_descriptions = [
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