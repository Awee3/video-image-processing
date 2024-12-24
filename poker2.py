import pygame
import sys
import os
import cv2
import tensorflow as tf
import numpy as np
import logging
import random
from itertools import combinations
from collections import Counter

pygame.init()

SCREEN_WIDTH, SCREEN_HEIGHT = 1600, 900
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("POKER")

MAIN_MENU = pygame.transform.scale(pygame.image.load(
    os.path.join('assets', 'bg3.png')), (SCREEN_WIDTH, SCREEN_HEIGHT))
GAME_BG = pygame.transform.scale(pygame.image.load(
    os.path.join('assets', 'bg1.png')), (SCREEN_WIDTH, SCREEN_HEIGHT))

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GOLD = (255, 215, 0)

class Button():
    def __init__(self, x, y, text, font, text_color, bg_color, width, height):
        self.x = x
        self.y = y
        self.text = text
        self.font = font
        self.text_color = text_color
        self.bg_color = bg_color
        self.width = width
        self.height = height
        self.rect = pygame.Rect(x, y, width, height)
        self.clicked = False

    def draw(self, surface):
        # Draw the button background
        pygame.draw.rect(surface, self.bg_color, self.rect)

        # Render the text
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

        # Check for mouse events
        action = False
        pos = pygame.mouse.get_pos()

        if self.rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0] == 1 and not self.clicked:
                self.clicked = True
                action = True

        if pygame.mouse.get_pressed()[0] == 0:
            self.clicked = False

        return action

font = pygame.font.Font(None, 70)  
font2= pygame.font.Font(None, 100)
font3= pygame.font.Font(None, 50)
font4= pygame.font.Font(None, 30)

judul_text = font2.render("POKER TEXAS", True, GOLD)
preFlop_text = font3.render("PreFlop Round", True, GOLD)
flop_text = font3.render("Flop Round", True, GOLD)
turn_text = font3.render("Turn Round", True, GOLD)
river_text = font3.render("River Round", True, GOLD)

play_button = Button(200, 550, "PLAY", font, WHITE, BLACK, 200, 100)
exit_button = Button(200, 700, "EXIT", font, WHITE, BLACK, 200, 100)
start_button = Button(1300, 300, "START", font, WHITE, BLACK, 200, 100)
next_button = Button(1400, 650, "NEXT", font, WHITE, BLACK, 150, 100)
mainmenu_button = Button(1400,770, "EXIT", font, WHITE, BLACK, 150, 100)

call_button_p1 = Button(500, 700, "CALL", font3, WHITE, BLACK, 130, 50)
raise_button_p1 = Button(500, 780, "RAISE", font3, WHITE, BLACK, 130, 50)
call_button_p2 = Button(500, 50, "CALL", font3, WHITE, BLACK, 130, 50)
raise_button_p2 = Button(500, 130, "RAISE", font3, WHITE, BLACK, 130, 50)

middle_card1 = Button(500, 400, "", font, WHITE, BLACK, 100,140)
middle_card2 = Button(620, 400, "", font, WHITE, BLACK, 100,140)
middle_card3 = Button(740, 400, "", font, WHITE, BLACK, 100,140)
middle_card4 = Button(860, 400, "", font, WHITE, BLACK, 100,140)
middle_card5 = Button(980, 400, "", font, WHITE, BLACK, 100,140)

player1_card1 = Button(700, 700, "", font, WHITE, BLACK, 100,140)
player1_card2 = Button(820, 700, "", font, WHITE, BLACK, 100,140)

player2_card1 = Button(700, 50, "", font, WHITE, BLACK, 100,140)
player2_card2 = Button(820, 50, "", font, WHITE, BLACK, 100,140)

def main_menu():
    while True:
        screen.blit(MAIN_MENU, (0,0))
        if play_button.draw(screen):
            print("Play button clicked!") 
            break
        if exit_button.draw(screen):
            pygame.quit()
            sys.exit()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.flip()
    main()

def main():
    running = True
    clock = pygame.time.Clock()
    round = 0
    is_flop = False
    is_turn = False
    is_river = False
    pot = 0  
    player1_chip = 2000  
    player2_chip = 2000  
    current_bet = 100  
    pot_text = font3.render(f"Pot: {pot}", True, GOLD)
    player1_chip_text = font3.render(f"Chips: {player1_chip}", True, GOLD)
    player2_chip_text = font3.render(f"Chips: {player2_chip}", True, GOLD)
    player1_action = False
    player2_action = False

    class_mapping_file = 'model/class_mapping.txt'
    class_names = load_class_mapping(class_mapping_file)
    model = tf.keras.models.load_model('model/64x3-cards.h5')
    cap = cv2.VideoCapture(2)
    frame_width = 100 
    frame_height = 200
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    predictions = []
    prediction_image = None

    while running:
        screen.blit(GAME_BG, (0, 0))
        screen.blit(player1_chip_text, (930, 700))
        screen.blit(player2_chip_text, (930, 50))
        screen.blit(pot_text, (740, 350))
       
        player1_card1.draw(screen)
        player1_card2.draw(screen)
        player2_card1.draw(screen)
        player2_card2.draw(screen)
        middle_card1.draw(screen)
        middle_card2.draw(screen)
        middle_card3.draw(screen)
        middle_card4.draw(screen)
        middle_card5.draw(screen)

        current_bet_text = font3.render(f"Current Bet: {current_bet}", True, GOLD)
        screen.blit(current_bet_text, (100, 50)) 

        if call_button_p1.draw(screen):  
            if player1_chip >= current_bet:
                player1_chip -= current_bet
                pot += current_bet
                player1_action = True  
                print("Player 1 Call: 100")
                pot_text = font3.render(f"Pot: {pot}", True, GOLD)
                player1_chip_text = font3.render(f"Chips: {player1_chip}", True, GOLD)

        if raise_button_p1.draw(screen): 
            raise_amount = 100  # Kelipatan raise
            new_bet = current_bet + raise_amount 
            if player1_chip >= current_bet:
                player1_chip -= new_bet
                pot += new_bet
                current_bet = new_bet
                player1_action = True 
                print("Player 1 Raise: 200")
                pot_text = font3.render(f"Pot: {pot}", True, GOLD)
                player1_chip_text = font3.render(f"Chips: {player1_chip}", True, GOLD)

        if call_button_p2.draw(screen):  # Player 2 Call
            if player2_chip >= current_bet:
                player2_chip -= current_bet
                pot += current_bet
                player2_action = True  
                pot_text = font3.render(f"Pot: {pot}", True, GOLD)
                player2_chip_text = font3.render(f"Chips: {player2_chip}", True, GOLD)

        if raise_button_p2.draw(screen): 
            raise_amount = 100  # Kelipatan raise
            new_bet = current_bet + raise_amount 
            if player2_chip >= current_bet:
                player2_chip -= new_bet
                pot += new_bet
                current_bet = new_bet
                player2_action = True  
                print("Player 2 Raise: 200")
                pot_text = font3.render(f"Pot: {pot}", True, GOLD)
                player2_chip_text = font3.render(f"Chips: {player2_chip}", True, GOLD)

        if player1_action and player2_action:  
            player1_action = False  
            player2_action = False
            round += 1
            if round == 2:  
                print("Flop")
                screen.blit(flop_text, (1300, 40))
                is_flop = True
            elif round == 3:  # Flop ke Turn
                print("Turn")
                screen.blit(turn_text, (1300, 40))
                is_turn = True
            elif round == 4:  # Turn ke River
                print("River")
                screen.blit(river_text, (1300, 40))
                is_river = True
                
        if round == 1:
            screen.blit(preFlop_text,(1300,40))
        if round == 2:
            screen.blit(flop_text,(1300,40))
        if round == 3:
            screen.blit(turn_text,(1300,40))
        if round == 4:
            screen.blit(river_text,(1300,40))
            best_deck_info = best_deck(player1_deck, dealer_deck, player2_deck)
            winner_text = font4.render(f"Winner: {best_deck_info['winner']}", True, GOLD) 
            winning_combination_text = font4.render(f"Winning Combination: {best_deck_info['winning_combination_type']}", True, GOLD)
            player1_text = font4.render(f"Player 1 Best Hand: {best_deck_info['player1']['best_combination_type']} ({best_deck_info['player1']['best_value']})", True, GOLD)
            player2_text = font4.render(f"Player 2 Best Hand: {best_deck_info['player2']['best_combination_type']} ({best_deck_info['player2']['best_value']})", True, GOLD)
            screen.blit(winner_text, (1200, 300)) 
            screen.blit(winning_combination_text, (1200, 350))
            screen.blit(player1_text, (1200, 400))
            screen.blit(player2_text, (1200, 450))

            winner = best_deck_info['winner']
            if winner == "Player 1":
                print("Player 1 Takes all")
                player1_chip += pot
                pot = 0
            elif winner == "Player 2":
                print("Player 2 Takes all")
                player2_chip += pot
                pot = 0
            elif winner == "Draw":
                print("no one take the moneyy")
                player1_chip = 2000
                player2_chip = 2000
                pot = 0
            player1_chip_text = font3.render(f"Chips: {player1_chip}", True, GOLD)
            player2_chip_text = font3.render(f"Chips: {player2_chip}", True, GOLD)
            pot_text = font3.render(f"Pot: {pot}", True, GOLD)
            
        for i, img in enumerate(player1_images):
            screen.blit(pygame.transform.scale(img, (100,140)), player1_positions[i])

        for i, img in enumerate(player2_images):
            screen.blit(pygame.transform.scale(img, (100,140)), player2_positions[i])

        if is_flop :
            for i, img in enumerate(dealer_images):
                if i < 3:
                    screen.blit(pygame.transform.scale(img, (100, 140)), dealer_positions[i])

        if is_turn :
            if len(dealer_images) >= 4: 
                screen.blit(pygame.transform.scale(dealer_images[3], (100, 140)), dealer_positions[3])

        if is_river :
            if len(dealer_images) >= 5:  
                screen.blit(pygame.transform.scale(dealer_images[4], (100, 140)), dealer_positions[4])

        #-----fungsi deteksi kartu-----
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera")
            break

        # Konversi ke HSV dan buat mask untuk mendeteksi warna biru
        kernel = np.ones((3, 3), np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([120 - 20, 50, 50])
        upper_blue = np.array([120 + 20, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_inv = cv2.bitwise_not(mask)
        mask = cv2.erode(mask, kernel, iterations=4)
        mask = cv2.dilate(mask, kernel, iterations=4)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            x, y,w,h = cv2.boundingRect(contour)
            if len(approx) == 4:
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                for point in approx:
                    cv2.circle(frame, tuple(point[0]), 5, (0, 0, 255), -1)
                pts_original = np.float32([point[0] for point in approx])
                pts_original = sorted(pts_original, key=lambda x: (x[1], x[0]))
                pts_original = np.float32([pts_original[0], pts_original[1], pts_original[2], pts_original[3]])
                width, height = 128,128  
                pts_transformed = np.float32([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]])
                matrix = cv2.getPerspectiveTransform(pts_original, pts_transformed)
                output_warped = cv2.warpPerspective(frame, matrix, (width, height))
                # resized_warped = cv2.resize(output_warped, (128, 128))  # Resize to (128, 128)
                resized_warped = cv2.cvtColor(output_warped, cv2.COLOR_BGR2GRAY)
                resized_warped = resized_warped.astype(np.float32) / 255.0  # Scale pixel values to [0, 1]
                prediction = model.predict(resized_warped[np.newaxis, ...])  # Add batch dimension
                predicted_class = np.argmax(prediction)  # Get the class with the highest probability
                class_name = class_names.get(predicted_class, "Unknown") if class_names else "Unknown"
                cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                predictions.append((class_name))   
        new_width, new_height = 300, 300 
        resized_frame = cv2.resize(frame, (new_width, new_height))
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB) 
        frame_surface = pygame.image.frombuffer(resized_frame.tobytes(), (new_width, new_height), 'RGB')
        # frame_surface = pygame.image.frombuffer(frame.tobytes(), (frame.shape[1], frame.shape[0]), 'BGR')
        screen.blit(frame_surface, (10, 250))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                key = event.key
                if key == pygame.K_s:  
                    if predictions:
                        predicted_card = predictions[-1]
                        print("Hasil prediksi disimpan:", predicted_card) 

                        if predicted_card in player1_deck or predicted_card in player2_deck or predicted_card in dealer_deck:
                            print(f"Kartu '{predicted_card}' sudah ada, coba kartu lain.")
                        else:
                            try:
                                image_path = os.path.join('images', f"{predicted_card}.jpg")
                                prediction_image = pygame.image.load(image_path)
                                print(f"Gambar '{image_path}' dimuat.")
                            
                                if len(player1_images) < 2 :
                                    player1_images.append(prediction_image)
                                    player1_deck.append(predicted_card)
                                elif len(player2_images) <2 :
                                    player2_images.append(prediction_image)
                                    player2_deck.append(predicted_card)
                                elif len(dealer_images) < 5 :
                                    dealer_images.append(prediction_image)
                                    dealer_deck.append(predicted_card)
                    
                                if len(dealer_images) == 5 and len(player1_images) == 2 and len(player2_images) == 2 :
                                    round = 1
                                    
                            except FileNotFoundError:
                                print(f"Gambar untuk '{predicted_card}' tidak ditemukan.")
                    elif key == pygame.K_q:
                        cap.release()         
        #------------------------------------------------------

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
  
        clock.tick(30)

def load_class_mapping(mapping_file):
    if os.path.exists(mapping_file):
        class_names = {}
        with open(mapping_file, 'r') as f:
            for line in f:
                index, name = line.strip().split(': ')
                class_names[int(index)] = name
        return class_names
    else:
        print(f"Mapping file '{mapping_file}' not found.")
        return None

dealer_positions = [(500, 400), (620,400), (740, 400),(860, 400),(980, 400)]  
player1_positions = [(700, 700), (820,700)] 
player2_positions = [(700, 50), (820,50)] 

dealer_images = []  
player1_images = []  
player2_images = []

card_values = {
    "ace of clubs": [1, 14],
    "ace of diamonds": [1, 14],
    "ace of hearts": [1, 14],
    "ace of spades": [1, 14],
    "two of clubs": 2,
    "two of diamonds": 2,
    "two of hearts": 2,
    "two of spades": 2,
    "three of clubs": 3,
    "three of diamonds": 3,
    "three of hearts": 3,
    "three of spades": 3,
    "four of clubs": 4,
    "four of diamonds": 4,
    "four of hearts": 4,
    "four of spades": 4,
    "five of clubs": 5,
    "five of diamonds": 5,
    "five of hearts": 5,
    "five of spades": 5,
    "six of clubs": 6,
    "six of diamonds": 6,
    "six of hearts": 6,
    "six of spades": 6,
    "seven of clubs": 7,
    "seven of diamonds": 7,
    "seven of hearts": 7,
    "seven of spades": 7,
    "eight of clubs": 8,
    "eight of diamonds": 8,
    "eight of hearts": 8,
    "eight of spades": 8,
    "nine of clubs": 9,
    "nine of diamonds": 9,
    "nine of hearts": 9,
    "nine of spades": 9,
    "ten of clubs": 10,
    "ten of diamonds": 10,
    "ten of hearts": 10,
    "ten of spades": 10,
    "jack of clubs": 11,
    "jack of diamonds": 11,
    "jack of hearts": 11,
    "jack of spades": 11,
    "queen of clubs": 12,
    "queen of diamonds": 12,
    "queen of hearts": 12,
    "queen of spades": 12,
    "king of clubs": 13,
    "king of diamonds": 13,
    "king of hearts": 13,
    "king of spades": 13
}

all_cards = list(card_values.keys())

def get_card_suit(card_name):
    if " of " in card_name:
        return card_name.split(" of ")[1].lower()  # ngambil bagian setelah "of"
    return None 

dealer_deck = []
player1_deck = []
player2_deck = []
dealer_total = 0
player_total = 0

def evaluate_deck(deck):
    values = []
    for card in deck:
        card_value = card_values[card]
        values.append(card_value if isinstance(card_value, int) else max(card_value))
    suits = [card.split(" of ")[1] for card in deck]  
    
    value_counts = Counter(values)
    most_common = value_counts.most_common()  
    is_flush = len(set(suits)) == 1  
    sorted_values = sorted(values)
    is_straight = all(sorted_values[i] + 1 == sorted_values[i + 1] for i in range(len(sorted_values) - 1))
    
    # kalo ace straight dari bawah jd 1, kalo dari atas jadi 14
    if not is_straight and set([14, 2, 3, 4, 5]).issubset(values):
        is_straight = True
        sorted_values = [5, 4, 3, 2, 1]

    # combo
    if is_flush and set([10, 11, 12, 13, 14]).issubset(values):
        return "Royal Flush", 10
    
    if is_flush and is_straight:
        return "Straight Flush", 9
    elif most_common[0][1] == 4:
        return "Four of a Kind", 8
    elif most_common[0][1] == 3 and most_common[1][1] == 2:
        return "Full House", 7
    elif is_flush:
        return "Flush", 6
    elif is_straight:
        return "Straight", 5
    elif most_common[0][1] == 3:
        return "Three of a Kind", 4
    elif most_common[0][1] == 2 and most_common[1][1] == 2:
        return "Two Pair", 3
    elif most_common[0][1] == 2:
        return "One Pair", 2
    else:
        return "High Card", 1
    
def best_deck(player1_deck, dealer_deck, player2_deck):
    combined_cards_player1 = player1_deck + dealer_deck
    combined_cards_player2 = player2_deck + dealer_deck

    best_combination_player1 = None
    best_deck_value_player1 = None
    best_combination_type_player1 = None

    #p1
    for deck in combinations(combined_cards_player1, 5):
        combination_type, deck_value = evaluate_deck(deck)  # Menggunakan fungsi evaluate_deck
        if best_deck_value_player1 is None or deck_value > best_deck_value_player1:
            best_combination_player1 = deck
            best_deck_value_player1 = deck_value
            best_combination_type_player1 = combination_type

    best_combination_player2 = None
    best_deck_value_player2 = None
    best_combination_type_player2 = None

    #p2
    for deck in combinations(combined_cards_player2, 5):
        combination_type, deck_value = evaluate_deck(deck)  # Menggunakan fungsi evaluate_deck
        if best_deck_value_player2 is None or deck_value > best_deck_value_player2:
            best_combination_player2 = deck
            best_deck_value_player2 = deck_value
            best_combination_type_player2 = combination_type

    # cek winner
    if best_deck_value_player1 > best_deck_value_player2:
        winner = "Player 1"
        winning_combination = best_combination_player1
        winning_combination_type = best_combination_type_player1
        winning_value = best_deck_value_player1
    elif best_deck_value_player2 > best_deck_value_player1:
        winner = "Player 2"
        winning_combination = best_combination_player2
        winning_combination_type = best_combination_type_player2
        winning_value = best_deck_value_player2
    else:
        winner = "Draw"
        winning_combination = None
        winning_combination_type = "Equal"  
        winning_value = best_deck_value_player1 

    return {
        "winner": winner,
        "winning_combination": winning_combination,
        "winning_combination_type": winning_combination_type,
        "winning_value": winning_value,
        "player1": {
            "best_combination": best_combination_player1,
            "best_combination_type": best_combination_type_player1,
            "best_value": best_deck_value_player1,
        },
        "player2": {
            "best_combination": best_combination_player2,
            "best_combination_type": best_combination_type_player2,
            "best_value": best_deck_value_player2,
        },
    }
    
if __name__ == "__main__":
    main_menu()