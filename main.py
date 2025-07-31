import cv2
import numpy as np
import random
import time

# Game state variables (used globally)
player_score = 0
computer_score = 0
round_count = 0
max_rounds = 5
show_round_start = True
round_start_time = 0
game_result = ""
player_move = ""
computer_move = ""


def get_hand_gesture(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return "None"

    max_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_contour) < 2000:
        return "None"

    hull = cv2.convexHull(max_contour, returnPoints=False)
    if len(hull) < 3:
        return "None"

    defects = cv2.convexityDefects(max_contour, hull)
    if defects is None:
        return "Rock"

    count_defects = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])

        a = np.linalg.norm(np.array(end) - np.array(start))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(end) - np.array(far))
        angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c + 1e-5))

        if angle <= np.pi / 2:
            count_defects += 1

    if count_defects == 0:
        return "Rock"
    elif count_defects == 1:
        return "Scissors"
    elif count_defects >= 3:
        return "Paper"
    else:
        return "None"


def get_winner(player, computer):
    if player == computer:
        return "Draw"
    elif (player == "Rock" and computer == "Scissors") or \
         (player == "Scissors" and computer == "Paper") or \
         (player == "Paper" and computer == "Rock"):
        return "Player"
    else:
        return "Computer"


def process_frame(frame):
    global player_score, computer_score, round_count, max_rounds
    global show_round_start, round_start_time, game_result
    global player_move, computer_move

    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    roi = frame[100:300, 100:300]

    current_time = time.time()

    if round_count < max_rounds:
        if show_round_start:
            round_start_time = current_time
            show_round_start = False
            player_move = ""
            computer_move = ""
            game_result = ""

        if current_time - round_start_time < 2:
            cv2.putText(frame, f"Round {round_count + 1} starting...", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        elif current_time - round_start_time < 5:
            gesture = get_hand_gesture(roi)
            if gesture != "None" and player_move == "":
                player_move = gesture
                computer_move = random.choice(["Rock", "Paper", "Scissors"])
                winner = get_winner(player_move, computer_move)

                if winner == "Player":
                    player_score += 1
                    game_result = "You win this round!"
                elif winner == "Computer":
                    computer_score += 1
                    game_result = "Computer wins this round!"
                else:
                    game_result = "It's a draw!"

                round_count += 1
        else:
            show_round_start = True

    # Display UI text
    cv2.putText(frame, f"Round: {round_count}/{max_rounds}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
    cv2.putText(frame, f"You: {player_score}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Computer: {computer_score}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if player_move:
        cv2.putText(frame, f"You chose: {player_move}", (10, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Computer: {computer_move}", (10, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, game_result, (10, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    if round_count == max_rounds:
        final_msg = "Game Over: "
        if player_score > computer_score:
            final_msg += "You Win the Game!"
        elif computer_score > player_score:
            final_msg += "Computer Wins the Game!"
        else:
            final_msg += "It's a Tie!"

        cv2.putText(frame, final_msg, (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    return frame
