import React from 'react';
import {useAuth} from "../AuthContext";
import { useRanger } from 'react-ranger'
import {Track, TickLabel, Tick, Segment, Handle} from "../App"


export default function JokeCard() {
    const {currentUser, signout, currentJoke, setRating, rating, rateJoke, loading} = useAuth();

    const { getTrackProps, ticks, segments, handles } = useRanger({
        min: -10,
        max: 10,
        stepSize: 1,
        values: rating,
        onChange: setRating
      });


    return (
        <div className="box">
            <h2>Your Humor</h2>
            <p>Welcome {currentUser.email}!</p>
            <p>{currentJoke[1]}</p>
            <h4>Rate the Joke:</h4>
            <div key={rating}>
            <Track {...getTrackProps()}>
                {ticks.map(({ value, getTickProps }) => (
                <Tick {...getTickProps()}>
                    <TickLabel>{value}</TickLabel>
                </Tick>
                ))}
                {segments.map(({ getSegmentProps }, i) => (
                <Segment {...getSegmentProps()} index={i} />
                ))}
                {handles.map(({ value, active, getHandleProps }) => (
                <button
                    {...getHandleProps({
                    style: {
                        appearance: "none",
                        border: "none",
                        background: "transparent",
                        outline: "none"
                    }
                    })}
                >
                    <Handle active={active}>{value}</Handle>
                </button>
                ))}
            </Track>
            </div>
            <br />
            <br />
            <button className="box-button" onClick={rateJoke} disabled={loading}>Get new joke!</button>
            <button className="box-button" onClick={signout}>Signout</button>
        </div>
    )
}