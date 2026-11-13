# Error analysis (test fold 5, 8 clips per class)

## Accuracy per ESC-50 group

| Group | CNN (mixup, seed 1) | CLAP probe | CLAP zero-shot |
|---|---|---|---|
| Animals | 0.81 | 0.97 | 0.82 |
| Natural soundscapes & water | 0.85 | 0.99 | 0.86 |
| Human, non-speech | 0.84 | 0.96 | 0.94 |
| Interior / domestic | 0.84 | 0.96 | 0.93 |
| Exterior / urban | 0.70 | 0.93 | 0.91 |

## CNN (mixup, seed 1): 77 errors out of 400

Hardest classes: airplane (0.12), hen (0.25), water_drops (0.38), pig (0.38), helicopter (0.38), fireworks (0.50), clapping (0.50), wind (0.62)

Most frequent confusions (true -> predicted, count):

- fireworks -> crackling_fire: 4
- airplane -> thunderstorm: 4
- wind -> siren: 3
- clapping -> rain: 3
- hen -> laughing: 3
- washing_machine -> helicopter: 2
- water_drops -> can_opening: 2
- hen -> crying_baby: 2
- airplane -> wind: 2
- pig -> rooster: 1

## CLAP probe: 15 errors out of 400

Hardest classes: helicopter (0.38), pig (0.75), laughing (0.75), breathing (0.88), can_opening (0.88), clock_tick (0.88), keyboard_typing (0.88), chirping_birds (0.88)

Most frequent confusions (true -> predicted, count):

- helicopter -> washing_machine: 2
- helicopter -> airplane: 2
- laughing -> crying_baby: 2
- pig -> rooster: 1
- pig -> door_wood_creaks: 1
- helicopter -> crickets: 1
- clock_tick -> mouse_click: 1
- keyboard_typing -> mouse_click: 1
- engine -> washing_machine: 1
- breathing -> wind: 1

## CLAP zero-shot: 43 errors out of 400

Hardest classes: insects (0.12), wind (0.25), hen (0.38), helicopter (0.38), washing_machine (0.50), crickets (0.50), drinking_sipping (0.75), pig (0.75)

Most frequent confusions (true -> predicted, count):

- hen -> rooster: 5
- wind -> train: 4
- helicopter -> airplane: 4
- insects -> airplane: 4
- washing_machine -> airplane: 3
- wind -> airplane: 2
- insects -> cow: 2
- drinking_sipping -> pig: 2
- crickets -> frog: 2
- crickets -> insects: 2
