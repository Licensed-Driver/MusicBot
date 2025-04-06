import { useEffect } from 'react'

export const AuthCallback = () => {

    useEffect(() => {
        // Once the spotify auth redirects the user with the authorization code, we send that to the backend for secure processing of data
        const url = new URLSearchParams(window.location.search)
        const accessCode=url.get("code")

        if(accessCode) {
            fetch("/api/userdata",{
                method:'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({code:accessCode})
            })
            .then(response => response.json())
            .then(data => window.postMessage(data, window.location.origin))
        } else {
            console.log("Failed to get access code.")
            window.postMessage({access_granted:false, message:'Couldn\'t Extract Code'}, window.location.origin)
        }
    })

    return(<p>Logging you in...</p>)
}