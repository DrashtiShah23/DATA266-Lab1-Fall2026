# Task 1 failure case candidates

These three candidates are the saved continuations with the highest within-sample repeated 4-gram rates. The category below is the mechanical observation from that rate. The written explanation is left for the student.

Checkpoint for every candidate: `task1_llm/drashti/checkpoints/task1_drashti_train_20260928T070850656698Z_epoch10_step15630.pt`

## Candidate 1

Prompt:

```text
Bob was a big pupil. His teacher said he
```

Generated text:

```text
Bob was a big pupil. His teacher said he was so happy. He was so excited to go on an adventure. He was so excited to go on an adventure. He was so excited to go on an adventure. He was so excited to g
```

- Checkpoint: `task1_llm/drashti/checkpoints/task1_drashti_train_20260928T070850656698Z_epoch10_step15630.pt`
- Generation method: greedy
- Temperature: not used
- Maximum generated characters: 160
- Actual new characters: 160
- Within-sample repeated 4-gram rate: 0.6878980891719745
- Observable candidate category: repetition

The model starts fine (“he was so happy… excited to go on an adventure”) and then gets stuck. The same sentence fragment is copied almost verbatim for the rest of the 160 characters. That is classic greedy collapse: once “He was so excited to go on an adventure” becomes high probability, every next step prefers to keep that loop. The repeated 4-gram rate of 0.69 is not an abstract metric here—you can see the loop by eye. I classify this as repetition, not just “boring writing,” because the failure mode is local n-gram lock-in rather than a wrong plot choice.

## Candidate 2

Prompt:

```text
Sara and Ben went to the zoo with their 
```

Generated text:

```text
Sara and Ben went to the zoo with their mom. They saw a big box of cars and trucks. They wanted to buy some cars and trucks. They wanted to see the box and started to buy some food. They were so excit
```

- Checkpoint: `task1_llm/drashti/checkpoints/task1_drashti_train_20260928T070850656698Z_epoch10_step15630.pt`
- Generation method: greedy
- Temperature: not used
- Maximum generated characters: 160
- Actual new characters: 160
- Within-sample repeated 4-gram rate: 0.35668789808917195
- Observable candidate category: loss of coherence
- Visible text: the continuation leaves the zoo and describes a box of cars and trucks, then buying food.

The prompt sets a zoo scene, but the continuation never stays there. Mom appears (reasonable), then “a big box of cars and trucks,” then buying food—none of which follows from “went to the zoo.” This is loss of coherence: the model is stitching together TinyStories-ish templates (box of toys, buying food, excitement) without holding the topic. Character-level next-token loss can still be low while the story jumps, because each short phrase is locally plausible.

## Candidate 3

Prompt:

```text
Lily and Ben are playing in their room. 
```

Generated text:

```text
Lily and Ben are playing in their room. They like to run and jump and slide. They see a big box of toys. They want to see the box and the box and the box and the box was very happy.

"Let's go to the 
```

- Checkpoint: `task1_llm/drashti/checkpoints/task1_drashti_train_20260928T070850656698Z_epoch10_step15630.pt`
- Generation method: greedy
- Temperature: not used
- Maximum generated characters: 160
- Actual new characters: 160
- Within-sample repeated 4-gram rate: 0.3057324840764331
- Observable candidate category: repetition

After a normal opening, the text hits “They want to see the box and the box and the box and the box was very happy.” That is the same repetition failure as Candidate 1, just with a different phrase. The model treats “the box and” as a safe continuation and rolls it until the character budget ends. I keep the category as repetition; the grammar break at the end (“the box was very happy”) is a side effect of that loop, not the main story.
