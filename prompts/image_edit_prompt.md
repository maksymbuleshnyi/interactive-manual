You are writing an image editing prompt. The editor will modify the CURRENT
image to depict the NEXT state. Apply the minimum visual change required.

Current image description: {{current_state_description}}
Next state description:    {{next_state_description}}
User instruction:          {{natural_language_instruction}}

Rules:
- State the single change clearly: what is added, moved, or altered.
- If the changed object moves from one place to another, state both its new
  position and that it is no longer in its previous position.
- Add a short clause telling the editor to leave the rest of the image
  unchanged.
- If any object near or similar to the changed one could be wrongly moved or
  removed, name it explicitly and say it stays in its place.
- Describe the changed element as it appears in the result, not the action.
- Be concrete about the changed object's position and visible attributes.

Example of the expected shape (change + moved-from note, then "leave the rest
unchanged", then any at-risk object named as staying put):
"Attach the flat-blade North American plug head to the white power adapter; it
is now on the adapter and no longer lying on the desk. Leave the rest of the
image unchanged, and keep the other, differently-shaped plug head in its place
on the desk."

Return: a single paragraph image-edit prompt, no preamble.
