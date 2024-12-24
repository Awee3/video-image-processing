import pygame
import sys
import os
import cv2
import tensorflow as tf
import numpy as np

pygame.init()

SCREEN_WIDTH, SCREEN_HEIGHT = 1430, 953
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Main Menu")

MAIN_MENU = pygame.transform.scale(pygame.image.load(
    os.path.join('assets', 'bg2.png')), (SCREEN_WIDTH, SCREEN_HEIGHT))
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

font = pygame.font.Font(None, 74)  
font2= pygame.font.Font(None, 100)
# Create button
judul_text = font2.render("BLACKJACK", True, GOLD)
dealer_text = font.render("Dealer's Card", True, GOLD)
player_text = font.render("Player's Card", True, GOLD)
dealer_score = font.render("SCORE :", True, GOLD)
player_score = font.render("SCORE :", True, GOLD)

play_button = Button(615, 350, "PLAY", font, WHITE, BLACK, 200, 100)
exit_button = Button(615, 500, "EXIT", font, WHITE, BLACK, 200, 100)
start_button = Button(200, 350, "START", font, WHITE, BLACK, 200, 100)
mainmenu_button = Button(200, 500, "EXIT", font, WHITE, BLACK, 200, 100)


dealer_card1 = Button(700, 50, "", font, WHITE, BLACK, 150, 250)
dealer_card2 = Button(900, 50, "", font, WHITE, BLACK, 150, 250)
dealer_card3 = Button(1100, 50, "", font, WHITE, BLACK, 150, 250)

player_card1 = Button(700, 650, "", font, WHITE, BLACK, 150, 250)
player_card2 = Button(900, 650, "", font, WHITE, BLACK, 150, 250)
player_card3 = Button(1100, 650, "", font, WHITE, BLACK, 150, 250)

def main_menu():
    while True:
        screen.blit(MAIN_MENU, (0,0))
        screen.blit(judul_text,(510,150))
        # Draw buttons
        if play_button.draw(screen):
            print("Play button clicked!")  # Replace with your game start function
            break
        if exit_button.draw(screen):
            pygame.quit()
            sys.exit()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.flip()

    main()

def main():
    running = True
    clock = pygame.time.Clock()

    while running:
        screen.blit(GAME_BG, (0, 0))
        screen.blit(dealer_text,(200,250))
        screen.blit(player_text,(200,650))
        dealer_card1.draw(screen)
        dealer_card2.draw(screen)
        dealer_card3.draw(screen)
        player_card1.draw(screen)
        player_card2.draw(screen)
        player_card3.draw(screen)

        # Tampilkan gambar kartu dealer
        for i, img in enumerate(dealer_images):
            screen.blit(pygame.transform.scale(img, (150, 250)), dealer_positions[i])

        # Tampilkan gambar kartu player
        for i, img in enumerate(player_images):
            screen.blit(pygame.transform.scale(img, (150, 250)), player_positions[i])

        scores()
        if len(dealer_images) == 3 and len(player_images) == 3:
            check_winner()

        if start_button.draw(screen):
            print("Start button clicked!")
            deal_card()  

        if mainmenu_button.draw(screen):
            print("Start button clicked!")
            main_menu()
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        
        clock.tick(30)

# Function to load class name mapping
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

dealer_positions = [(700, 50), (900, 50), (1100, 50)]  
player_positions = [(700, 650), (900, 650), (1100, 650)] 

# Daftar untuk menyimpan gambar prediksi
dealer_images = []  # Gambar untuk dealer
player_images = []  # Gambar untuk player
card_order = ["dealer", "player", "dealer", "player", "dealer", "player"]  
card_index = 0  

def deal_card():
    global card_index
    card_owner = card_order[card_index]
    # Fungsi detect_card akan ditentukan berdasarkan pemilik kartu (dealer atau player)
    detect_card(card_owner)  
    
    # Update index untuk urutan kartu selanjutnya
    card_index = (card_index + 1) % len(card_order) 

def detect_card(card_owner):
    class_mapping_file = 'model/class_mapping.txt'
    class_names = load_class_mapping(class_mapping_file)

    model = tf.keras.models.load_model('model/cnn_model_v7.h5')
 
    cap = cv2.VideoCapture(1)
    predictions = []
    prediction_image = None
    while True:
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
            x, y, w, h = cv2.boundingRect(contour)
            if len(approx) == 4:
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                for point in approx:
                    cv2.circle(frame, tuple(point[0]), 5, (0, 0, 255), -1)

                pts_original = np.float32([point[0] for point in approx])
                pts_original = sorted(pts_original, key=lambda x: (x[1], x[0]))
                pts_original = np.float32([pts_original[0], pts_original[1], pts_original[2], pts_original[3]])
                width, height = 200, 300  
                pts_transformed = np.float32([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]])
                matrix = cv2.getPerspectiveTransform(pts_original, pts_transformed)
                output_warped = cv2.warpPerspective(frame, matrix, (width, height))
                resized_warped = cv2.resize(output_warped, (128, 128))  # Resize to (128, 128)
                resized_warped = resized_warped.astype(np.float32) / 255.0  # Scale pixel values to [0, 1]
                prediction = model.predict(resized_warped[np.newaxis, ...])  # Add batch dimension
                predicted_class = np.argmax(prediction)  # Get the class with the highest probability
                class_name = class_names.get(predicted_class, "Unknown") if class_names else "Unknown"
                cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                predictions.append((class_name))
        cv2.imshow("Label", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if predictions:
                predicted_card = predictions[-1]
                print("Hasil prediksi disimpan:", predictions[-1]) 
                try:
                    image_path = os.path.join('images', f"{predictions[-1]}.jpg")
                    prediction_image = pygame.image.load(image_path)
                    print(f"Gambar '{image_path}' dimuat.")
                    
                    if card_owner == "dealer" and len(dealer_images) < 3:
                        dealer_images.append(prediction_image)
                        dealer_deck.append(predicted_card)
                    elif card_owner == "player" and len(player_images) < 3:
                        player_images.append(prediction_image)
                        player_deck.append(predicted_card)

                    global dealer_total, player_total
                    dealer_total = calculate_deck_value(dealer_deck)
                    player_total = calculate_deck_value(player_deck)

                    pygame.display.flip()
                except FileNotFoundError:
                    print(f"Gambar untuk '{predictions[-1]}' tidak ditemukan.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

card_values = {
    "ace of clubs": [1, 11],
    "ace of diamonds": [1, 11],
    "ace of hearts": [1, 11],
    "ace of spades": [1, 11],
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
    "jack of clubs": 10,
    "jack of diamonds": 10,
    "jack of hearts": 10,
    "jack of spades": 10,
    "queen of clubs": 10,
    "queen of diamonds": 10,
    "queen of hearts": 10,
    "queen of spades": 10,
    "king of clubs": 10,
    "king of diamonds": 10,
    "king of hearts": 10,
    "king of spades": 10
}

dealer_deck = []
player_deck = []
dealer_total = 0
player_total = 0

def calculate_deck_value(deck):
    total = 0
    aces = 0

    for card in deck:
        value = card_values.get(card, 0)  
        if isinstance(value, list):  
            aces += 1
            total += value[1]  
        else:
            total += value

    
    while total > 21 and aces > 0:
        total -= 10  
        aces -= 1

    return total

def scores():
    player_score_text = font.render(f"SCORE: {player_total}", True, GOLD)
    dealer_score_text = font.render(f"SCORE: {dealer_total}", True, GOLD)
    screen.blit(dealer_score_text, (200, 150))
    screen.blit(player_score_text, (200, 750))

def check_winner():
    if player_total > 21:
        player_bust = font.render("Dealer wins! Player busted", True, GOLD)
        screen.blit(player_bust, (700, 450))
        return "Dealer wins! Player busted."
    elif dealer_total > 21:
        dealer_bust = font.render("Player wins! Dealer busted", True, GOLD)
        screen.blit(dealer_bust, (700, 450))
        return "Player wins! Dealer busted."
    elif player_total > dealer_total:
        player_win = font.render("Player wins!", True, GOLD)
        screen.blit(player_win, (700, 450))
        return "Player wins!"
    elif player_total < dealer_total:
        dealer_win = font.render("Dealer wins!", True, GOLD)
        screen.blit(dealer_win, (700, 450))
        return "Dealer wins!"
    else:
        return "It's a tie!"

if __name__ == "__main__":
    main_menu()