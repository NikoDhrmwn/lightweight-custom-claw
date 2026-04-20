---
name: discord-interactive-chat
description: Use this skill whenever the user asks for interactive Discord chat behavior such as buttons, choice prompts, lightweight polls, pick-one messages, or branching follow-up replies after someone clicks an option. Triggers include mentions of Discord buttons, interactive messages, choose/select between options, polls, or wanting the bot to respond differently based on what someone picked.
license: MIT
---

# Discord interactive chat

Use this skill when the current task is about making Discord conversations interactive instead of plain text only.

## What this skill is for

- Sending a short prompt with clickable choices.
- Letting people pick between up to 5 options.
- Posting a predefined follow-up reply after a user clicks one of the options.
- Keeping the public chat flow playful and fast.

## Primary tool

Use `send_interactive_choices` when you need real Discord buttons.

Parameters:
- `prompt`: the message shown above the buttons
- `options`: an array of 1 to 5 short labels
- `responses`: an optional object mapping each option label to the public reply that should be sent when that option is clicked

## Good patterns

- Keep button labels very short, ideally 1 to 3 words.
- Make the prompt feel natural and chatty.
- If the user wants a roast, joke, or reaction for each option, put that directly into `responses`.
- When the interactive message itself already fulfills the request, do not over-explain afterward.
- If the user asks for a poll or quick choice, prefer buttons over a long normal reply.

## Constraints

- Buttons are best for up to 5 choices.
- Each response should be self-contained because it will be sent later, after the click.
- Do not promise dropdowns, modals, or multi-step UI if the available tool only supports buttons.
- If the channel does not support interactive choices, say so plainly and fall back to normal text.

## Example

If the user says:

`make an interactive message to choose between tofu or tempeh, then roast them for the choice`

Use:

```json
{
  "prompt": "Tofu or tempeh? Pick one.",
  "options": ["Tofu", "Tempeh"],
  "responses": {
    "Tofu": "Tofu? Bold move. You picked the soft-spoken cube with main-character delusions.",
    "Tempeh": "Tempeh? You picked the aggressively textured one. Respectfully, that is a very committed personality choice."
  }
}
```

Then keep any extra explanation brief.
