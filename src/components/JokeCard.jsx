import React from 'react';
import {useAuth} from "../AuthContext";
// import Select from "react-select";
import { useRanger } from 'react-ranger'
import {Track, TickLabel, Tick, Segment, Handle} from "../App"


export default function JokeCard() {
    const {currentUser, signout, currentJoke, setRating, rating, rateJoke, loading} = useAuth();
    // const ratings = [
    //     { label: "+10", value: 10 },
    //     { label: "+9", value: 9 },
    //     { label: "+8", value: 8 },
    //     { label: "+7", value: 7 },
    //     { label: "+6", value: 6 },
    //     { label: "+5", value: 5 },
    //     { label: "+4", value: 4 },
    //     { label: "+3", value: 3 },
    //     { label: "+2", value: 2 },
    //     { label: "+1", value: 1 },
    //     { label: "0", value: 0 },
    //     { label: "-1", value: -1 },
    //     { label: "-2", value: -2 },
    //     { label: "-3", value: -3 },
    //     { label: "-4", value: -4 },
    //     { label: "-5", value: -5 },
    //     { label: "-6", value: -6 },
    //     { label: "-7", value: -7 },
    //     { label: "-8", value: -8 },
    //     { label: "-9", value: -9 },
    //     { label: "-10", value: -10 },
    //   ];
    
    // const colourStyles = {
    //     option: (styles, { data, isDisabled, isFocused, isSelected }) => {
    //         return {
    //         ...styles,
    //         backgroundColor: isFocused ? "#d6a9e8" : null,
    //         color: "#333333"
    //         };
    //     }
    // };

    // const handler = (event) => {
    //     const value = event.value
    //     // console.log(value)
    //     setRating(value)
    // }

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
            {/* <div key={rating}>
            <Select
            placeholder={rating}
            options={ ratings }
            onChange={handler}
            styles={colourStyles}/>
        </div> */}
            <h4>Rate the Joke</h4>
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
            {/* </div> */}
        </div>
    )
}