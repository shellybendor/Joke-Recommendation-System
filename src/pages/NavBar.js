import React, {useState} from "react";
// import {
//     Navbar,
//     Nav,
//     Button,
//     Alert
// } from "react-bootstrap";
import {useAuth} from "../AuthContext";


export default function NavBar() {
    const {currentUser, signout} = useAuth();


    return (

        <nav>
        {currentUser &&
                        <button onClick={signout}>Signout</button>
                    }
        </nav>

    );
}