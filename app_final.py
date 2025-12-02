from flask import Flask, request, jsonify, render_template
import numpy as np
import random

app = Flask(__name__)

# Enhanced story-based dialogues with more variations
STORY_DIALOGUES = {
    'romance': [
        ("I never believed in love at first sight until I saw you across the room", "From that moment, I knew my life would never be the same"),
        ("I have to tell you something I have been keeping inside", "Whatever it is, you can tell me. I am here for you"),
        ("I got the job in Paris, but I cannot go without you", "Paris? I have always dreamed of seeing Paris with you"),
        ("This ring belonged to my grandmother. She told me to give it to someone special", "Are you saying what I think you are saying?"),
        ("I have loved you since the day we met in the coffee shop", "That was three years ago. Why did you never say anything?"),
        ("Every time I see you, my heart skips a beat", "I feel the same way about you. You complete me"),
        ("I want to spend the rest of my life with you", "That is all I have ever wanted to hear"),
        ("You make me a better person just by being in my life", "And you make me believe in happy endings"),
        ("I never thought I could feel this way about anyone", "We were meant to find each other in this big world"),
        ("Will you marry me and make me the happiest person alive?", "Yes! A thousand times yes!")
    ],
    'thriller': [
        ("The files are missing. Someone has been in the system", "I told you we should have increased security. This is bad"),
        ("There is a bomb in the building. We have twenty minutes to find it", "I know where it is. Follow me to the basement"),
        ("The witness is dead. Someone got to him before we could", "This goes deeper than we thought. We are being watched"),
        ("I found the encrypted message. It leads to the old warehouse", "Do not go alone. Take backup. This could be a trap"),
        ("The target has escaped. He knows we are after him", "He cannot get far. We have every exit covered"),
        ("The security cameras are down. Someone is inside the building", "We need to evacuate immediately. This is not a drill"),
        ("I decoded the message. It is a list of targets and we are on it", "We need to warn the others before it is too late"),
        ("The evidence points to someone inside our organization", "A mole? That explains how they always know our moves"),
        ("They have taken the ambassador. We have six hours to get him back", "I know their patterns. We can intercept them at the border"),
        ("The virus has been activated. We are locked in the building", "There is a backup system in the emergency room. We can override it")
    ],
    'comedy': [
        ("I accidentally adopted three dogs today. They were having a special", "Three dogs? We live in a studio apartment. This is going to be interesting"),
        ("I tried to cook dinner, but I think I set the kitchen on fire", "Again? This is the third time this month. Maybe we should order pizza"),
        ("I told my boss what I really think about his new haircut", "You did not. Please tell me you are joking. That is career suicide"),
        ("I entered us in a dance competition. The prize is ten thousand dollars", "But I have two left feet. This is going to be a disaster"),
        ("I may have accidentally booked our vacation in the wrong country", "How do you accidentally book tickets to Antarctica? That takes effort"),
        ("I thought the costume party was tomorrow so I went to work like this", "Dressed as a giant chicken? No wonder everyone was staring"),
        ("I just won a year supply of pickles in a radio contest", "What are we going to do with five hundred jars of pickles?"),
        ("I tried to fix the sink and now there is a fountain in the kitchen", "Maybe we should call a professional before the neighbors complain"),
        ("I signed us up for couple's therapy with my parents", "Your parents? This is either brilliant or the worst idea ever"),
        ("I may have adopted a raccoon thinking it was a cat", "That explains why it keeps trying to open the refrigerator")
    ],
    'drama': [
        ("The test results came back. It is not good news", "Whatever it is, we will face it together. You are not alone"),
        ("I lost everything in the stock market crash. Our savings are gone", "Money comes and goes. We still have each other. That is what matters"),
        ("I saw you with her yesterday. Who is she and why were you together", "It is not what you think. Let me explain everything"),
        ("I have decided to leave the company and start my own business", "That is a big risk, but I believe in you. We will make it work"),
        ("The doctor said it is a miracle you survived the accident", "I had to survive. I could not leave you and the children"),
        ("After all these years, I found my birth mother. She lives in the next town", "This changes everything. Do you want to meet her?"),
        ("I have been offered a job on the other side of the country", "What about us? What about everything we built here?"),
        ("The letter arrived today. The adoption finally went through", "After five years of waiting, we are finally going to be parents"),
        ("I have to tell you the truth about what happened that night", "I think I already know, but I need to hear it from you"),
        ("The war is over. He is coming home after three years", "I cannot believe it. I thought this day would never come")
    ],
    'action': [
        ("The building is surrounded. We need an escape plan now", "There is a service tunnel in the basement. It leads to the river"),
        ("The hostage situation is critical. We have to move in now", "Wait for my signal. We only get one chance at this"),
        ("The virus has been released. We have one hour to find the antidote", "I know where it is hidden. Follow me to the laboratory"),
        ("The enemy forces are advancing. We are outnumbered ten to one", "Then we make our stand here. They will not take this position"),
        ("The bomb is armed. Thirty seconds until detonation", "I can disarm it. Just keep them off my back for twenty seconds"),
        ("The helicopter is waiting on the roof. We have two minutes to extract", "I will cover you while you get the civilians to safety"),
        ("The bridge is about to collapse. We need to get everyone across now", "I will secure the other side. Move quickly but carefully"),
        ("They have taken the nuclear codes. We have to stop them before they reach the border", "I can intercept their convoy at the mountain pass"),
        ("The EMP blast took out all electronics. We are operating blind", "We have old school maps and compasses. We can still navigate"),
        ("The sniper has us pinned down. We cannot move from this position", "I see a ventilation shaft. I can flank them from above")
    ]
}

CHARACTER_PERSONALITIES = {
    'JOHN': {'style': 'direct', 'emotion': 'confident'},
    'MARY': {'style': 'caring', 'emotion': 'compassionate'},
    'DETECTIVE': {'style': 'suspicious', 'emotion': 'alert'},
    'SUSPECT': {'style': 'defensive', 'emotion': 'nervous'},
    'FRIEND1': {'style': 'humorous', 'emotion': 'playful'},
    'FRIEND2': {'style': 'sarcastic', 'emotion': 'amused'},
    'BOSS': {'style': 'authoritative', 'emotion': 'serious'},
    'EMPLOYEE': {'style': 'respectful', 'emotion': 'anxious'},
    'HERO': {'style': 'brave', 'emotion': 'determined'},
    'VILLAIN': {'style': 'menacing', 'emotion': 'confident'},
    'ROSS': {'style': 'analytical', 'emotion': 'anxious'},
    'RACHEL': {'style': 'emotional', 'emotion': 'dramatic'},
    'JOEY': {'style': 'simple', 'emotion': 'hungry'},
    'MONICA': {'style': 'controlling', 'emotion': 'stressed'},
    'CHANDLER': {'style': 'sarcastic', 'emotion': 'nervous'},
    'PHOEBE': {'style': 'quirky', 'emotion': 'mystical'}
}

# Additional dialogue elements to increase length
FOLLOW_UP_DIALOGUES = {
    'romance': [
        "I never want this moment to end",
        "You mean everything to me",
        "This feels like a dream come true",
        "I have been waiting my whole life for this",
        "You make me believe in destiny",
        "My heart belongs to you forever",
        "I cannot imagine my life without you",
        "You are my everything",
        "This love is worth fighting for",
        "We were meant to be together"
    ],
    'thriller': [
        "We need to move quickly",
        "Time is running out",
        "They are getting closer",
        "We cannot trust anyone",
        "The stakes are too high",
        "We are running out of options",
        "This changes everything",
        "We need a new plan",
        "They are one step ahead",
        "We are in grave danger"
    ],
    'comedy': [
        "This keeps getting better and better",
        "I cannot believe this is happening",
        "What else could possibly go wrong?",
        "This is the story of our lives",
        "We will laugh about this someday",
        "Only you could get into this situation",
        "This is absolutely ridiculous",
        "I should have seen this coming",
        "Well, that did not go as planned",
        "This is going to be a great story"
    ],
    'drama': [
        "This changes everything",
        "We need to think about this carefully",
        "The consequences are serious",
        "This is a turning point for us",
        "We cannot go back from this",
        "This decision will define us",
        "We have to face the truth",
        "Nothing will ever be the same",
        "We need to be strong now",
        "This is our reality now"
    ],
    'action': [
        "We have to move now",
        "There is no time to waste",
        "The mission comes first",
        "We cannot afford mistakes",
        "This is our only chance",
        "We are running out of time",
        "The situation is critical",
        "We need to improvise",
        "Failure is not an option",
        "We have to succeed"
    ]
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_script():
    try:
        data = request.json
        characters = data.get('characters', ['JOHN', 'MARY'])
        genre = data.get('genre', 'romance')
        max_length = data.get('max_length', 200)
        
        print(f"Generating story script for: {characters}, genre: {genre}, target length: {max_length} words")
        
        # Generate story-driven script that respects length
        script = generate_story_driven_script(characters, genre, max_length)
        
        actual_word_count = len(script.split())
        
        return jsonify({
            'success': True,
            'script': script,
            'features': f"Generated {actual_word_count} words in {genre} genre (target: {max_length})"
        })
        
    except Exception as e:
        print(f"Error generating script: {e}")
        return jsonify({'error': str(e)}), 500

def generate_story_driven_script(characters, genre, target_word_count):
    """Generate a story-driven script that respects the target word count"""
    script_lines = []
    current_word_count = 0
    
    # Start with a story setup
    setups = {
        'romance': f"{characters[0]} and {characters[1]} are having a heartfelt conversation",
        'thriller': f"{characters[0]} discovers something alarming and confronts {characters[1]}",
        'comedy': f"{characters[0]} gets into an unexpected situation and tells {characters[1]} about it",
        'drama': f"{characters[0]} faces a life-changing decision and discusses it with {characters[1]}",
        'action': f"{characters[0]} and {characters[1]} are in a high-stakes situation"
    }
    
    scene_header = f"SCENE START - {setups.get(genre, setups['romance'])}"
    script_lines.append(scene_header)
    script_lines.append("")
    current_word_count += len(scene_header.split())
    
    available_stories = STORY_DIALOGUES.get(genre, STORY_DIALOGUES['romance'])
    follow_ups = FOLLOW_UP_DIALOGUES.get(genre, FOLLOW_UP_DIALOGUES['romance'])
    
    # Calculate how many story arcs we can include based on target length
    # Each story arc typically adds 30-50 words
    words_per_arc = 40
    max_arcs = max(1, (target_word_count - current_word_count) // words_per_arc)
    
    print(f"Target: {target_word_count} words, Planning {max_arcs} story arcs")
    
    used_stories = set()
    scene_count = 0
    
    while current_word_count < target_word_count * 0.8 and scene_count < max_arcs * 2:
        if len(used_stories) >= len(available_stories):
            # If we've used all stories, start recycling but add variations
            used_stories.clear()
        
        # Choose an unused story arc
        available_indices = [i for i in range(len(available_stories)) if i not in used_stories]
        if not available_indices:
            available_indices = list(range(len(available_stories)))
        
        story_idx = random.choice(available_indices)
        used_stories.add(story_idx)
        story_arc = available_stories[story_idx]
        
        # Get character personalities
        char1_personality = CHARACTER_PERSONALITIES.get(characters[0], {'style': 'neutral', 'emotion': 'neutral'})
        char2_personality = CHARACTER_PERSONALITIES.get(characters[1], {'style': 'neutral', 'emotion': 'neutral'})
        
        # First character speaks
        line1 = adapt_dialogue_to_character(story_arc[0], char1_personality)
        script_lines.append(f"{characters[0]}: {line1}")
        current_word_count += len(line1.split()) + 1  # +1 for character name
        
        # Second character responds
        line2 = adapt_dialogue_to_character(story_arc[1], char2_personality)
        script_lines.append(f"{characters[1]}: {line2}")
        current_word_count += len(line2.split()) + 1
        
        # Add follow-up conversation to increase length naturally
        if current_word_count < target_word_count * 0.7 and random.random() > 0.3:
            # Character 1 follow-up
            follow_up1 = random.choice(follow_ups)
            adapted_follow1 = adapt_dialogue_to_character(follow_up1, char1_personality)
            script_lines.append(f"{characters[0]}: {adapted_follow1}")
            current_word_count += len(adapted_follow1.split()) + 1
            
            # Character 2 response
            follow_up2 = random.choice(follow_ups)
            adapted_follow2 = adapt_dialogue_to_character(follow_up2, char2_personality)
            script_lines.append(f"{characters[1]}: {adapted_follow2}")
            current_word_count += len(adapted_follow2.split()) + 1
        
        script_lines.append("")
        current_word_count += 1  # Empty line
        scene_count += 1
        
        print(f"Progress: {current_word_count}/{target_word_count} words after {scene_count} scenes")
    
    # If we're still under target, add more dialogue
    while current_word_count < target_word_count * 0.9:
        char_idx = random.randint(0, len(characters) - 1)
        character = characters[char_idx]
        personality = CHARACTER_PERSONALITIES.get(character, {'style': 'neutral', 'emotion': 'neutral'})
        
        additional_line = random.choice(follow_ups)
        adapted_line = adapt_dialogue_to_character(additional_line, personality)
        script_lines.append(f"{character}: {adapted_line}")
        current_word_count += len(adapted_line.split()) + 1
        
        # Occasionally add a response
        if random.random() > 0.5:
            responder_idx = (char_idx + 1) % len(characters)
            responder = characters[responder_idx]
            responder_personality = CHARACTER_PERSONALITIES.get(responder, {'style': 'neutral', 'emotion': 'neutral'})
            
            response_line = random.choice(follow_ups)
            adapted_response = adapt_dialogue_to_character(response_line, responder_personality)
            script_lines.append(f"{responder}: {adapted_response}")
            current_word_count += len(adapted_response.split()) + 1
        
        script_lines.append("")
        current_word_count += 1
    
    script_lines.append("SCENE END")
    current_word_count += 2
    
    final_script = '\n'.join(script_lines)
    actual_word_count = len(final_script.split())
    
    print(f"Final script: {actual_word_count} words (target: {target_word_count})")
    
    return final_script

def adapt_dialogue_to_character(dialogue, personality):
    """Adapt dialogue to match character personality"""
    style = personality['style']
    emotion = personality['emotion']
    
    # Simple adaptations based on style
    if style == 'sarcastic' and random.random() > 0.5:
        if '.' in dialogue:
            dialogue = dialogue.replace('.', '... obviously')
        if '!' in dialogue:
            dialogue = dialogue.replace('!', '... not!')
    elif style == 'humorous' and random.random() > 0.6:
        dialogue = dialogue + " ... funny, right?"
    elif style == 'emotional' and random.random() > 0.5:
        if '.' in dialogue:
            dialogue = dialogue.replace('.', '... and I mean it!')
    elif style == 'analytical' and random.random() > 0.4:
        dialogue = dialogue + " ... if you think about it logically."
    elif style == 'quirky' and random.random() > 0.5:
        dialogue = dialogue.replace('the', 'the cosmic').replace('a', 'a mystical')
    
    return dialogue

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    print("Starting Advanced Story-Driven Script Generator...")
    print("This version properly scales script length based on your input")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)






