# Heygen Dailyco Pipecat Example

Prototype integrating a [Heygen](https://heygen.com/) live-streaming avatar with [dailyco](https://www.daily.co/) for real-time video communication, powered by [pipecat](https://github.com/pipecat-ai/pipecat) for AI-driven media routing. The setup works well, successfully streaming the avatar feed, though there is a noticeable slow-motion, I will seek help from Heygen using this repository.

## Setup

```bash
make install-uv
```

Make sure you added uv binary to your path or run following command:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Run

Copy `.env.example` to `.env` and fill in the values. Then run:

```bash
make run
```

It will automatically open the room URL in your default browser.
