from __future__ import annotations

import random
from typing import Optional

from .contract import (
    FieldRole,
    ResponseControl,
    ResponseSpec,
    ResponseStep,
    Task,
    response_control_tokens,
)

TEMPLATES_PER_TASK = 30

TEMPLATES: dict[Task, tuple[str, ...]] = {
    Task.AUDIO_AR: (
        'Extend the {language} monologue.',
        'Add more {language} dialogue.',
        'Keep the {language} narrative going.',
        'Elaborate on the {language} topic.',
        'Expand the {language} narration.',
        'Continue speaking in {language}.',
        'Carry on the {language} conversation.',
        'Provide additional {language} commentary.',
        'Keep delivering in {language}.',
        'Continue the {language} dialogue.',
        'Build upon the {language} storyline.',
        'Keep the {language} discourse flowing.',
        'Add extra {language} remarks.',
        'Continue developing the {language} script.',
        'Proceed with more {language} narration.',
        'Extend the {language} speech section.',
        'Keep the {language} speech ongoing.',
        'Expand on the {language} dialogue.',
        'Add further {language} narration.',
        'Continue the {language} discourse without pause.',
        'Keep producing more {language} utterances.',
        'Add new {language} sentences.',
        'Expand the {language} storyline further.',
        'Keep the {language} flow going.',
        'Write more {language} commentary.',
        'Continue the {language} monologue further.',
        'Extend the {language} narrations.',
        'Keep adding {language} speech.',
        'Generate more {language} audio.',
        'Keep speaking naturally in {language}.',
    ),
    Task.ASR: (
        'Convert the {language} audio to text: {source}',
        'Capture the {language} speech in {source}',
        'Record the {language} speech from {source}',
        'Write down the {language} speech: {source}',
        'Map the {language} speech to text using {source}',
        'Document the {language} speech: {source}',
        'Speak and transcribe the {language} content: {source}',
        'Listen to {source} and transcribe the {language}',
        'Write a transcript of the {language} speech: {source}',
        'Note the {language} speech content from {source}',
        'Capture {language} speech via {source}',
        'Convert {language} speech from {source} to text',
        'Record the {language} audio signal in {source}',
        'Write a transcript for {language} speech: {source}',
        'Transcribe the {language} dialogue in {source}',
        'Copy down the {language} speech: {source}',
        'Document the {language} spoken words: {source}',
        'Capture {language} vocalization from {source}',
        'Convert {language} language-specific speech in {source}',
        'Record and process {language} speech via {source}',
        'Write the {language} speech recording: {source}',
        'Transcribe the {language} message contained in {source}',
        'Transcribe informal {language} speech from {source}',
        'Capture {language} tone and transcribe: {source}',
        'Document {language} speech samples: {source}',
        'Convert {language} voice data to text: {source}',
        'Record the {language} conversation in {source}',
        'Write down {language} pronunciation: {source}',
        'Transcribe {language} accented speech: {source}',
        'Produce a clean {language} transcript of {source}',
    ),
    Task.INTERLEAVED_AR: (
        'Continue the interleaved {language} text and speech.',
        'Keep alternating {language} text with speech.',
        'Extend the mixed {language} text-speech sequence.',
        'Continue the {language} interleaved response.',
        'Produce more interleaved {language} text and audio.',
        'Keep the {language} text-speech alternation going.',
        'Continue generating interleaved {language} content.',
        'Extend this {language} interleaved transcript.',
        'Add the next interleaved {language} text and speech chunks.',
        'Keep producing {language} interleaved tokens.',
        'Continue the {language} multimodal interleaved stream.',
        'Extend the interleaved {language} dialogue.',
        'Keep alternating spoken and written {language} content.',
        'Continue the {language} interleaved narrative.',
        'Generate more interleaved {language} turns.',
        'Keep the interleaved {language} sequence flowing.',
        'Continue with interleaved {language} text and speech.',
        'Extend the {language} interleaved output.',
        'Add further interleaved {language} content.',
        'Keep generating interleaved {language} material.',
        'Continue the interleaved {language} conversation.',
        'Produce the next interleaved {language} chunks.',
        'Keep going with interleaved {language} text and audio.',
        'Extend this interleaved {language} response.',
        'Continue alternating {language} modalities.',
        'Generate additional interleaved {language} content.',
        'Keep the {language} interleaved format going.',
        'Continue the mixed {language} interleaved sequence.',
        'Add more interleaved {language} text and speech.',
        'Proceed with interleaved {language} generation.',
    ),
    Task.MASKED_AR: (
        'Reconstruct masked {language} text and speech.',
        'Recover hidden {language} multimodal tokens.',
        'Fill masked spans in {language} text and audio.',
        'Restore masked {language} wording and speech.',
        'Predict held-out {language} text and audio tokens.',
        'Complete masked {language} multimodal content.',
        'Recover deleted {language} text and speech units.',
        'Unmask {language} transcript and audio tokens.',
        'Reconstruct missing {language} multimodal spans.',
        'Fill gaps in {language} text and speech.',
        'Restore masked {language} dual-modality tokens.',
        'Recover masked halves of {language} text and audio.',
        'Predict masked {language} lexical and acoustic tokens.',
        'Complete the masked {language} multimodal sequence.',
        'Reconstruct occluded {language} text and speech.',
        'Recover randomly masked {language} modalities.',
        'Fill masked {language} text-audio positions.',
        'Restore held-out {language} multimodal units.',
        'Unmask and reconstruct {language} speech-text.',
        'Predict missing {language} text and audio parts.',
        'Recover masked {language} continuation spans.',
        'Reconstruct masked {language} parallel tokens.',
        'Fill masked {language} interleaved content.',
        'Restore occluded {language} wording and audio.',
        'Recover masked {language} semantic and lexical tokens.',
        'Complete masked {language} multimodal MAE targets.',
        'Reconstruct held-out {language} text-audio tokens.',
        'Unmask {language} dual stream content.',
        'Predict masked {language} multimodal labels.',
        'Restore randomly dropped {language} text and speech.',
    ),
    Task.MT: (
        'Translate the following into {language}: {source}',
        'Render the text into {language}: {source}',
        'Convert the next phrase into {language}: {source}',
        'Provide a translation into {language} for this sentence: {source}',
        'Turn the following content into {language}: {source}',
        'Convert the given text to {language}: {source}',
        'Translate the next text into {language}: {source}',
        'Render this passage into {language}: {source}',
        'Provide a new version in {language} for {source}',
        'Turn the following passage into {language}: {source}',
        'Convert the following sentence into {language}: {source}',
        'Translate the enclosed material into {language}: {source}',
        'Render the paragraph in {language}: {source}',
        'Turn the given text into {language}: {source}',
        'Convert this text into {language}: {source}',
        'Translate the next passage into {language}: {source}',
        'Render the following text in {language}: {source}',
        'Provide a translation of {source} in {language}',
        'Turn this sentence into {language}: {source}',
        'Convert the following passage into {language}: {source}',
        'Translate the enclosed passage into {language}: {source}',
        'Render this text into {language}: {source}',
        'Provide a translation for {source} in {language}',
        'Turn the following into {language}: {source}',
        'Convert the following text into {language}: {source}',
        'Translate the next phrase into {language}: {source}',
        'Render the paragraph into {language}: {source}',
        'Provide a translation in {language} for {source}',
        'Turn the next material into {language}: {source}',
        'Give the {language} meaning of this text: {source}',
    ),
    Task.PARALLEL_AR: (
        'Continue the parallel {language} text and speech tracks.',
        'Keep generating parallel {language} text and audio.',
        'Extend both {language} text and speech streams.',
        'Continue the {language} parallel multimodal output.',
        'Produce more parallel {language} text and speech.',
        'Keep the parallel {language} tracks going.',
        'Continue generating parallel {language} content.',
        'Extend this {language} parallel response.',
        'Add the next parallel {language} text and speech spans.',
        'Keep producing parallel {language} tokens.',
        'Continue the {language} parallel multimodal stream.',
        'Extend the parallel {language} dialogue.',
        'Keep writing and speaking {language} in parallel.',
        'Continue the {language} parallel narrative.',
        'Generate more parallel {language} text and audio.',
        'Keep the parallel {language} sequence flowing.',
        'Continue with parallel {language} text and speech.',
        'Extend the {language} parallel output.',
        'Add further parallel {language} content.',
        'Keep generating parallel {language} material.',
        'Continue the parallel {language} conversation.',
        'Produce the next parallel {language} spans.',
        'Keep going with parallel {language} text and audio.',
        'Extend this parallel {language} response.',
        'Continue both {language} modalities together.',
        'Generate additional parallel {language} content.',
        'Keep the {language} parallel format going.',
        'Continue the mixed {language} parallel sequence.',
        'Add more parallel {language} text and speech.',
        'Proceed with parallel {language} generation.',
    ),
    Task.S2ST: (
        'Translate the speech into {language}: {source}',
        'Convert the following into {language} speech: {source}',
        'Render this speech in {language}: {source}',
        'Reproduce the dialogue in {language}: {source}',
        'Translate the audio into {language}: {source}',
        'Transform the speech to {language} words: {source}',
        'Speak the following in {language}: {source}',
        'Transcribe the speech into {language}: {source}',
        'Convert the utterance to {language}: {source}',
        'Translate this dialogue into {language}: {source}',
        'Reword the speech in {language}: {source}',
        'Translate the message into {language}: {source}',
        'Render the audio in {language}: {source}',
        'Output the speech in {language}: {source}',
        'Translate this text into {language}: {source}',
        'Convert the following utterance to {language}: {source}',
        'Translate the spoken content into {language}: {source}',
        'Render the dialogue in {language}: {source}',
        'Transcribe the audio into {language}: {source}',
        'Translate the following narration into {language}: {source}',
        'Convert the speech into {language} audio: {source}',
        'Translate the speech to {language} words: {source}',
        'Produce the speech in {language}: {source}',
        'Render this speech into {language}: {source}',
        'Translate the dialog into {language} format: {source}',
        'Please produce spoken output in {language}: {source}',
        'Translate this recording into {language}: {source}',
        'Take the spoken text and output it in {language}: {source}',
        'Transcribe the dialogue to {language}: {source}',
        'Speak the same content in {language}: {source}',
    ),
    Task.S2TT: (
        'Convert the spoken utterance to {language}: {source}',
        'Render the audio as {language} text: {source}',
        'Provide a {language} transcription of the speech: {source}',
        'Produce the {language} translation of the audio: {source}',
        'Give the {language} text version of the utterance: {source}',
        'Transform the speech into {language} written form: {source}',
        'Render the spoken words in {language}: {source}',
        'Deliver a {language} transcript of the audio: {source}',
        'Output the {language} translation for the speech: {source}',
        'Transcribe the audio into {language}: {source}',
        'Provide {language} text for the spoken input: {source}',
        'Convert the speech to {language} written output: {source}',
        'Produce a {language} rendition of the utterance: {source}',
        'Give the {language} written translation of the audio: {source}',
        'Render the utterance as {language} text: {source}',
        'Translate the spoken content into {language}: {source}',
        'Supply the {language} transcription of the speech: {source}',
        'Turn the audio into {language} written form: {source}',
        'Deliver the {language} text equivalent of the utterance: {source}',
        'Produce the {language} version of the speech: {source}',
        'Convert the utterance to {language} text: {source}',
        'Render the spoken words as {language} transcription: {source}',
        'Provide a {language} translation for the audio: {source}',
        'Output {language} text from the speech: {source}',
        'Transcribe the utterance into {language}: {source}',
        'Give the {language} written output for the audio: {source}',
        'Transform the speech into {language} transcript: {source}',
        'Deliver a {language} text rendering of the utterance: {source}',
        'Produce {language} transcription from the speech: {source}',
        'Write the {language} meaning of this speech: {source}',
    ),
    Task.TEXT_AR: (
        'Keep writing the text below.',
        'Proceed with the text provided.',
        'Elaborate on the text shown.',
        'Extend the text that follows.',
        'Carry on with the text given.',
        'Add more to the text above.',
        'Write the next part of this text.',
        'Finish the text presented here.',
        'Expand the text displayed below.',
        'Continue writing from this text.',
        'Keep going with the text provided.',
        'Proceed writing the text shown.',
        'Elaborate further on the text below.',
        'Extend the text that is given.',
        'Carry on writing the text above.',
        'Add to the text that follows.',
        'Write the continuation of this text.',
        'Finish writing the text presented.',
        'Expand the text displayed here.',
        'Continue with the text provided below.',
        'Keep writing the text shown above.',
        'Proceed with the text given here.',
        'Elaborate on the text displayed below.',
        'Extend the text presented above.',
        'Carry on with the text that follows.',
        'Add more to the text given.',
        'Write the next section of this text.',
        'Finish the text shown below.',
        'Expand the text provided here.',
        'Write the next sentences naturally.',
    ),
    Task.T2ST: (
        'Convert the following text into {language} speech: {source}',
        'Generate {language} speech from the text: {source}',
        'Produce the {language} speech version of: {source}',
        'Please render the given text as {language} speech: {source}',
        "Translate the text '{source}' into {language} speech.",
        'Speak the following text in {language}: {source}',
        "Voice the text '{source}' in {language}.",
        "Convert '{source}' into {language} speech.",
        "Render the text '{source}' into {language} speech.",
        "Produce {language} speech from '{source}'.",
        'Please vocalize the text: {source} into {language}.',
        'Articulate the following text in {language}: {source}',
        "Convert the source text '{source}' into {language} speech.",
        'Generate {language} speech for the text: {source}.',
        "Deliver the text '{source}' as {language} speech.",
        "Transform the text '{source}' into {language} speech.",
        "Adapt the text '{source}' into {language} speech.",
        "Internationalize the text '{source}' into {language} speech.",
        "Narrate the text '{source}' in {language}.",
        "Convert '{source}' to {language} speech.",
        "Render '{source}' as {language} speech.",
        "Produce speech in {language} from '{source}'.",
        "Speak '{source}' in {language} speech.",
        "Voice '{source}' as {language} speech.",
        'Convert the following into {language} speech: {source}.',
        'Generate speech in {language} from the text: {source}.',
        "Please convert '{source}' into {language} speech.",
        "Translate '{source}' into {language} speech.",
        "Produce the {language} speech of '{source}'.",
        'Speak this text in {language}: {source}',
    ),
    Task.T2TT: (
        'Transform the following text into {language}: {source}',
        'Convert this passage into {language}: {source}',
        'Render the following content into {language}: {source}',
        'Provide a translation of the text into {language}: {source}',
        'Translate the given text into {language}: {source}',
        'Rewrite this text in {language}: {source}',
        'Restate the following text in {language}: {source}',
        'Convert the text below into {language}: {source}',
        'Turn the following text into {language}: {source}',
        'Produce a translation into {language} for: {source}',
        'Translate the passage into {language}: {source}',
        'Express the following in {language}: {source}',
        'Render this into {language}: {source}',
        'Reformulate the text into {language}: {source}',
        'Translate the content below into {language}: {source}',
        'Convert the text into {language}: {source}',
        'Translate the words below into {language}: {source}',
        'Recast the following text in {language}: {source}',
        'Translate the message into {language}: {source}',
        'Put the following text into {language}: {source}',
        'Translate the given passage into {language}: {source}',
        'Turn the text into {language}: {source}',
        'Translate what is written below into {language}: {source}',
        'Reframe the following in {language}: {source}',
        'Translate the text beneath into {language}: {source}',
        'Convert the words into {language}: {source}',
        'Rewrite the passage in {language}: {source}',
        'Translate the sentence into {language}: {source}',
        'Render the text in {language}: {source}',
        'Give this text in {language}: {source}',
    ),
    Task.TTS: (
        'Convert {source} into spoken language.',
        'Synthesize audio from {source}.',
        'Generate speech based on {source}.',
        'Translate {source} into audio output.',
        'Produce voiceover for {source}.',
        'Render speech using {source} and {language}.',
        'Transform {source} into audible form.',
        'Create spoken representation of {source}.',
        'Audio-generate content from {source}.',
        'Voice-synthesize the text {source}.',
        'Convert {source} to a spoken format.',
        'Build a TTS response for {source}.',
        'Output spoken version of {source}.',
        'Generate voice clip from {source}.',
        'Produce narration using {source}.',
        'Synthesize speech with {source} as input.',
        'Render {source} as spoken text.',
        'Generate audio narration from {source}.',
        'Convert {source} into spoken words.',
        'Create speech synthesis from {source}.',
        'Generate spoken output for {source} using {language}.',
        'Synthesize voice from the text {source}.',
        'Produce spoken rendition of {source}.',
        'Generate audio from {source} text.',
        'Convert {source} into an utterance.',
        'Generate speech based on {source} input.',
        'Synthesize language audio for {source}.',
        'Render {source} into spoken format.',
        'Transform {source} into voice output.',
        'Speak the following text aloud: {source}',
    ),
    Task.TTS_VOICE_CLONE: (
        'Use the source speech voice to say this {language} text: {source}',
        'Speak this {language} text in the source voice: {source}',
        'Synthesize the {language} text with the source speaker: {source}',
        'Render this {language} text using the source speech voice: {source}',
        'Generate the target speech in the source voice: {source}',
        'Preserve the source speaker while saying in {language}: {source}',
        'Voice this {language} text like the source speech: {source}',
        'Produce {language} speech conditioned on the source voice: {source}',
        'Say this in {language} with the source speaker identity: {source}',
        'Create {language} speech matching the source voice: {source}',
        'Speak the target text with the source vocal identity: {source}',
        'Synthesize this in {language} from the source speech style: {source}',
        'Generate speech for this text using the source voice: {source}',
        'Render the target text in the source speaker style: {source}',
        'Use the provided speech voice for this {language} text: {source}',
        'Produce a {language} rendition in the source voice: {source}',
        'Keep the source speaker while speaking this text: {source}',
        'Generate the {language} audio with the source voice: {source}',
        'Synthesize the target utterance from the source voice: {source}',
        'Speak this target text as the source speaker: {source}',
        'Transfer the source voice to this {language} text: {source}',
        'Create the target speech with the source speaker: {source}',
        'Use the source vocal style to narrate in {language}: {source}',
        'Produce speech matching the source speaker for: {source}',
        'Render this text with the voice heard in the source speech: {source}',
        'Generate the target audio in the source speaker voice: {source}',
        'Say the following in {language} while preserving the source voice: {source}',
        'Synthesize this text with the source speech identity: {source}',
        'Voice the target utterance like the source speaker: {source}',
        'Speak this {language} text using the source audio as voice context: {source}',
    ),
}


def select_template(task: Task, index: Optional[int] = 0) -> str:
    values = TEMPLATES[task]
    if len(values) != TEMPLATES_PER_TASK:
        raise AssertionError(
            f"{task.value} template pool must provide exactly "
            f"{TEMPLATES_PER_TASK} templates, got {len(values)}."
        )
    resolved = _index(index, name=f"{task.value} template")
    if resolved is None:
        return random.choice(values)
    if resolved >= len(values):
        raise IndexError(
            f"template index {resolved} is outside the {task.value} pool "
            f"of size {len(values)}."
        )
    return values[resolved]


def evaluation_template_index(index: Optional[int]) -> int:
    """Fixed index for reproducible eval/generation (``null`` -> ``0``)."""
    resolved = _index(index, name="evaluation template")
    return 0 if resolved is None else resolved


def format_instruction(
    task: Task,
    *,
    source: str,
    language: str | None = None,
    index: Optional[int] = 0,
) -> str:
    resolved = _index(index, name=f"{task.value} template")
    if resolved is None:
        raise ValueError(
            f"format_instruction requires a fixed template index for {task.value}."
        )
    text = select_template(task, resolved)
    kwargs: dict[str, str] = {}
    if "{source}" in text:
        kwargs["source"] = source
    if "{language}" in text:
        if language is None:
            raise ValueError(
                f"{task.value} template requires a language placeholder value."
            )
        kwargs["language"] = language
    return text.format(**kwargs)


def format_response_instruction(
    instruction: str,
    response: ResponseSpec,
    *,
    language: str,
) -> str:
    """Append the exact typed response schema required by the task program."""
    if not isinstance(instruction, str) or not instruction:
        raise ValueError("response instruction base must be a non-empty string.")
    if not isinstance(response, ResponseSpec):
        raise TypeError("response instruction requires a ResponseSpec.")
    if not isinstance(language, str) or not language:
        raise ValueError("response instruction language must be a non-empty string.")
    controlled = any(
        step.control in {ResponseControl.ASR, ResponseControl.MT}
        for step in response.steps
    )
    if response.name == "direct" and not controlled:
        return instruction
    if response.name == "direct":
        return (
            instruction
            + "\nRespond in this exact format:\n"
            + _step_format(response.steps[0], language)
        )
    steps = [
        f"{index}. {_step_format(step, language)}"
        for index, step in enumerate(response.steps, start=1)
    ]
    return instruction + "\nRespond in this exact order:\n" + "\n".join(steps)


def _step_format(step: ResponseStep, language: str) -> str:
    description = _control_instruction(step, language)
    control = response_control_tokens(
        step.control,
        target_language=language,
    )
    if control is None:
        return description
    prefix = "".join(token.value for token in control.prefix)
    return f"{description} as {prefix}...{control.end.value}"


def _control_instruction(step: ResponseStep, language: str) -> str:
    if step.control is ResponseControl.ASR:
        if step.role is FieldRole.SOURCE:
            return "transcribe the source speech as text"
        return "transcribe the speech as text"
    if step.control is ResponseControl.MT:
        return f"produce the {language} translation as text"
    if step.control is ResponseControl.EOS:
        return "produce the text response"
    if step.control is ResponseControl.AUDIO:
        if step.role is FieldRole.TARGET:
            return f"generate the corresponding {language} speech"
        if step.role is FieldRole.SOURCE:
            return "reproduce the source audio"
    raise ValueError(f"unsupported response control: {step.control.value}.")


def _index(value: object, *, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or null.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


__all__ = [
    "TEMPLATES",
    "TEMPLATES_PER_TASK",
    "evaluation_template_index",
    "format_instruction",
    "format_response_instruction",
    "select_template",
]
