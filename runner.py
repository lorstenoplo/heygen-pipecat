import argparse
import os
import uuid

import aiohttp

from pipecat.transports.services.helpers.daily_rest import (
    DailyRESTHelper,
    DailyRoomParams,
    DailyRoomProperties,
)
from config import settings


async def configure(aiohttp_session: aiohttp.ClientSession):
    (url, token) = await configure_with_args(aiohttp_session)
    return (url, token)


async def configure_with_args(
    aiohttp_session: aiohttp.ClientSession,
    parser: argparse.ArgumentParser | None = None,
):
    daily_rest_helper = DailyRESTHelper(
        daily_api_key=settings.DAILYCO_API_KEY,
        daily_api_url=settings.DAILYCO_BASE_URL,
        aiohttp_session=aiohttp_session,
    )

    room = await daily_rest_helper.create_room(
        params=DailyRoomParams(
            name=str(uuid.uuid4()),
            properties=DailyRoomProperties(
                start_video_off=True,
                start_audio_off=True, # type: ignore
                max_participants=2,
            ),
        )
    )

    # Create a meeting token for the given room with an expiration 1 hour in
    # the future.
    expiry_time: float = 60 * 60

    token = await daily_rest_helper.get_token(room.url, expiry_time)

    return (room.url, token)
