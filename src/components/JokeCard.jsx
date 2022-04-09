import React from 'react';
import {useAuth} from "../AuthContext";


export default function JokeCard() {
    const {currentUser, signout, getJoke, currentJoke} = useAuth();

    return (
        <div className="box">
            <h2>Your Humor</h2>
            <p>Welcome {currentUser.email}!</p>
            <p>{currentJoke[1]}</p>
            <button onClick={getJoke}>Get new joke!</button>
            <button onClick={signout}>Signout</button>
        </div>
    )
}