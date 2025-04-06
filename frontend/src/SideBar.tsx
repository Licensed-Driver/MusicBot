import { useEffect, useState } from 'react'

type SidebarProps = {
    isOpen:boolean
    toggle: () => void
}

const height=600
const width=500
const left=(window.innerWidth/2)-(width/2)
const top=(window.innerHeight/2)-(height/2)

const client_id=import.meta.env.VITE_CLIENT_ID
const front_ngrok=import.meta.env.VITE_FRONT_NGROK
const redirect_uri=`${front_ngrok}/auth/callback`
const scope='playlist-read-private playlist-read-collaborative user-read-playback-state user-top-read user-read-recently-played user-library-read'

function SpotifyGetAuth({setNeedsLogin, needsLogin}:{setNeedsLogin:(value:boolean)=>void, needsLogin:boolean}) {
    const login_popup=window.open(
        `https://accounts.spotify.com/authorize?${new URLSearchParams({
            response_type: 'code',
            client_id:client_id,
            scope: scope,
            redirect_uri: redirect_uri,
        }).toString()}`,
        "Login With Spotify",
        `width=${width},height=${height},top=${top},left=${left}`
    )
    login_popup?.addEventListener("message", (event)=>{
        // If the listener was triggered by something that didn't come from our original popup window return
        if(event.origin !== window.location.origin) return;
        // Unpack the accessToken
        const { access_granted, message } = event.data
        // If the accessToken returned a usable token then we print our test
        if(access_granted) {
            console.log("Received Session ID:", message)
            localStorage.setItem('session_id', message)
            setNeedsLogin(!needsLogin)
        } else {
            console.log("Error in receiving token. Error message:", message)
            localStorage.removeItem('session_id')
        }

        login_popup.close()
    })
}

class profile_info {
    display_name:string
    profile_photo:{
        url:string,
        height:number,
        width:number
    }

    constructor() {
        this.display_name=''
        this.profile_photo={
            url:'',
            height:0,
            width:0
        }
    }
}

export function Sidebar(props:SidebarProps) {

    const [loggedIn, setLoggedIn] = useState(false)
    const [needsLogin, setNeedsLogin] = useState(false)
    const [profile, setProfile] = useState<profile_info>({
        display_name:'',
        profile_photo:{
            url:'',
            height:0,
            width:0
        }
    })

    useEffect(() => {
        // Assume profile information is now incorrect at load and if login state has changed
        localStorage.removeItem('profile')
        let try_session = localStorage.getItem('session_id')

        if(!try_session) {
            setLoggedIn(false)
            return
        }else if(try_session) {
            fetch('/api/userdata/profile', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_id: try_session})
            })
            .then(response => response.json())
            .then(data => {
                console.log(data)
                if(!data.error) {
                    localStorage.setItem('profile', JSON.stringify(data))
                    setProfile(data)
                    setLoggedIn(true)
                    return
                } else if(data.error=='Invalid Session ID') {
                    localStorage.removeItem('session_id')
                    setProfile(new profile_info())
                    setLoggedIn(false)
                    return
                } else if(data.error=='Untrusted IP') {

                } else if(data.error=='Not Authorized') {
                    SpotifyGetAuth({setNeedsLogin:setNeedsLogin, needsLogin:needsLogin})
                }
            })
        }
    }, [needsLogin])

    return(
        <div
            className={`relative transition-all duration-300 ${props.isOpen ? 'bg-zinc-900' : 'bg-none'} text-white ${props.isOpen ? 'shadow-lg' : 'shadow-none'}
            ${props.isOpen ? 'w-64' : 'w-16'} flex flex-col relative left-0 top-0 rounded-lg m-2 p-4`}
        >
            <button
                onClick={props.toggle}
                className={`${props.isOpen ? 'shadow-none' : 'shadow-lg'} p-2 m-2 bg-zinc-800 rounded hover:bg-zinc-700 focus:outline-zinc-500 absolute top-0 left-0 w-10 h-10`}
            >
                {props.isOpen ? '⮘' : '⮚'}
            </button>

            <button
            className={`${props.isOpen && !loggedIn ? 'relative pointer-events-auto opacity-100' : 'pointer-events-none opacity-0'} max-w-[30%] max-h-[40px] rounded-md text-green-400 bg-zinc-700 text-center top-0 left-[70%]`}
            onClick={()=>{
                if(!loggedIn) {
                    SpotifyGetAuth({setNeedsLogin, needsLogin})
                }
            }}
            >LOGIN</button>

            <img src={profile?.profile_photo.url || './assets/default-profile.png'} className={`${props.isOpen && loggedIn ? 'absolute pointer-events-auto opacity-100' : 'absolute pointer-events-none opacity-0'} w-[20%] aspect-square rounded-full top-[4%] left-[76%]`} />

            <nav className={`mt-10 space-y-4 w-full ${props.isOpen ? 'pointer-events-auto opacity-100' : 'pointer-events-none opacity-0'}`}>
                <a href="#" className="block text-sm hover:text-zinc-400">Home</a>
                <a href="#" className="block text-sm hover:text-zinc-400">Projects</a>
                <a href="#" className="block text-sm hover:text-zinc-400">Settings</a>
                <a href="#" className={`${loggedIn ? 'relative opacity-100 pointer-events-auto' : 'absolute opacity-0 pointer-events-none'} block text-sm hover:text-zinc-400`} onClick={() => {
                    localStorage.removeItem('session_id')
                    localStorage.removeItem('profile')
                    setNeedsLogin(!needsLogin)
                }}>Sign Out</a>
            </nav>
        </div>
    )
}
