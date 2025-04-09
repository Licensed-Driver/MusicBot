import { useEffect, useState, useRef } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { Html, useTexture } from '@react-three/drei'
//import html2canvas from 'html2canvas' // Part of the HTML to canvas and plane stuff that is useful but not being used
import './App.css'

import { Sidebar } from './SideBar'
import { AuthCallback } from './auth-callback'

import {SpaceInfo, AddSimpVec} from './helpers/Vector3'
import './workers/album_worker'
import './workers/title_worker'
const album_worker = new Worker(new URL('./workers/album_worker.ts', import.meta.url), {
  type: 'module',
})
const title_worker = new Worker(new URL('./workers/title_worker.ts', import.meta.url), {
  type: 'module'
})

// This is a function that converts HTML to a canvas texture but I'm not using it and Vite doesn't like that (I'll use it in the future)
/*const useDomToCanvas = (domEl:HTMLElement) => {

  // The debounce function just waits for the specified amount of time before executing the callback
  function debounce(func:Function, timeout:number) {
    let timer:number;
  
    return (...args:any) => {
      clearTimeout(timer)
      timer = setTimeout(() => {func.apply(args)}, timeout)
    }
  }

  const [texture, setTexture] = useState<THREE.CanvasTexture>();
  
  // Define all the getters and shit
  useEffect(() => {
    // Obviously a check
    if (!domEl) return;
    // Converts the html to a canvas on the side
    const convertDomToCanvas = async () => {
      const canvas = await html2canvas(domEl, { backgroundColor: null });
      setTexture(new THREE.CanvasTexture(canvas));
    };
    // Calls it
    convertDomToCanvas();

    // Define a callback to use for resizing the texture
    // Uses debounce since we don't wanna resize it immediately every update
    const debouncedResize = debounce(() => {
      convertDomToCanvas();
    }, 100);

    // Add listener for the resize
    window.addEventListener("resize", debouncedResize);

  }, [domEl]);

  return texture;
};*/

function RedirectHandler() {
  const location = useLocation()
  const allowedPath = '/spotifind'
  const spotifyCallback = '/auth/callback'

  if (!location.pathname.startsWith(allowedPath) || !location.pathname.startsWith(spotifyCallback)) {
    window.location.href = 'https://github.com/Licensed-Driver'
    return null // Don't render anything while redirecting
  }

  return <Navigate to={allowedPath} />
}

type song = {
  album: string,
  artists: string[],
  id: string,
  images: string,
  name: string
}

function Album({imgUrl, ...props}:{imgUrl:string}) {
    
  // Set up state for hovering and active states
  const [hovered, setHover] = useState(false)
  const [active, setActive] = useState(false)

  // Using a reference gives direct access to the mesh
  const meshRef = useRef<THREE.Mesh>(null!)
  // Using a reference to access the mesh material for updating the image
  const matRef = useRef<THREE.MeshBasicMaterial>(null!)
  // Lets us update the texture since it needs the texture to be constant and local
  const albumTex = useTexture(imgUrl, (texture) => {
    texture.minFilter = THREE.LinearFilter
    texture.magFilter = THREE.LinearFilter
    texture.anisotropy = 16 // better for sharpness at angles
    texture.generateMipmaps = true
    texture.needsUpdate = true
  })

  const { camera, mouse } = useThree();
  
  const [albumInfo, setAlbumInfo] = useState<SpaceInfo>({
    rotation:{x:0, y:0, z:0},
    position:{x:0, y:0, z:0},
    scale:{x:1, y:1, z:1}
  })

  const timer = useRef(0)

  useFrame((_, delta) => {
    // In case the mesh isn't set yet
    if(!meshRef.current) return

    timer.current += delta

    let CursorWorldPosition = AddSimpVec(camera.position, {x:mouse.x, y:mouse.y, z:camera.position.z})
    
    album_worker.postMessage({objectID:'ALBUM', type:'LOOK-AT', objectInfo:albumInfo, refPosition:CursorWorldPosition})

    // If the album is hovered over it'll expand over time for a smooth interaction    
    if(hovered) {
      album_worker.postMessage({objectID:'ALBUM', type:'HOVER', objectInfo:albumInfo,})
    } else album_worker.postMessage({objectID:'ALBUM', type:'UNHOVER', objectInfo:albumInfo})

    album_worker.onmessage = (e) => {
      const { objectID, objectInfo }:{objectID:string, objectInfo:SpaceInfo} = e.data
      if(objectID=='ALBUM') setAlbumInfo(objectInfo)
    }

  })

  return (
    <mesh
      {...props} // The properties passed automaticall
      ref={meshRef} // A reference to the object directly
      scale={[albumInfo.scale.x, albumInfo.scale.y, albumInfo.scale.z]} // Scales it up if it's active
      onClick={() => setActive(!active)} // Changes it's active state if it's clicked
      onPointerOver={() => setHover(true)} // This and one below track hovering
      onPointerOut={() => setHover(false)}
      position={[albumInfo.position.x, albumInfo.position.y, albumInfo.position.z]}
      rotation={[albumInfo.rotation.x, albumInfo.rotation.y, albumInfo.rotation.z]}
      >
      <planeGeometry args={[5, 5, 1]}/>
      <meshBasicMaterial ref={matRef} map={albumTex}/>
    </mesh>
  )
}

// This function returns a react object that uses a DOM node as a texture on a plane, but I'm not surrently using it
/*function domPlane({divRef, ...props}: {divRef:React.RefObject<HTMLElement>, props:any}) {
  if(divRef==null)return
  // Set a constant reference directly to the mesh for react to use
  const meshRef = useRef<THREE.Mesh>(null!)

  const [texture, setDomTexture] = useState<THREE.Texture | null>()

  useEffect(() => {
    setDomTexture(useDomToCanvas(divRef.current))
  }, [])

  return (
    <mesh
    {...props}
    ref={meshRef}
    >
    <planeGeometry args={[5,10]}/>
    <meshBasicMaterial map={texture}/>
    </mesh>
  )

}*/

function SearchUI3D({
  prediction,
  selectedSong,
  mousePosition
}: {
  prediction:number|null,
  selectedSong:song|null,
  mousePosition:{x:number, y:number}
}) {

  const { camera } = useThree();
  const [htmlPosition, setHtmlPosition] = useState({x:0, y:0})
  const [titleInfo, setTitleInfo] = useState<SpaceInfo>({
    rotation:{x:0, y:0, z:0},
    position:{x:0, y:0, z:0},
    scale:{x:1, y:1, z:1}
  })
  // For the reference to the actual Html drei element
  const htmlRef = useRef<HTMLDivElement>(null!)
  const [uiRect, setUIRect] = useState<DOMRect>(new DOMRect(0, 0, 0, 0))

  useEffect(() => {
    if(htmlRef.current) setUIRect(htmlRef.current.getBoundingClientRect())
  }, [htmlRef.current])

  useFrame(() => {

    // Gets the html position on the screen including when scrolling and turns it into a vector for ui worker to change the rotation to always follow the mouse
    // Divided by 500 to scale it down to a vector, but also to make the turning slight
    setHtmlPosition({x:uiRect.left + uiRect.width/2 + window.scrollX, y:uiRect.top + uiRect.height/2 + window.scrollY})
    const mouseVector = new THREE.Vector3((mousePosition.x - htmlPosition.x)/500, -(mousePosition.y - htmlPosition.y)/500, 0.5) // z=0.5 = mid-range depth
    mouseVector.unproject(camera) // converts camera position to world space
    title_worker.postMessage({objectID:'TITLE', type:'LOOK-AT', objectInfo:titleInfo, refPosition:mouseVector})

    title_worker.onmessage = (e) => {
      const { objectID, objectInfo }:{objectID:string, objectInfo:SpaceInfo} = e.data
      if(objectID=='TITLE') setTitleInfo(objectInfo)
    }
  });

  return (
    <>
      <Html ref={htmlRef} scale={[2, 2, 0.5]} position={[0,0,0]} rotation={[titleInfo.rotation.x, titleInfo.rotation.y, 0]} transform occlude
      style={{
        pointerEvents: 'none',
        willChange: 'transform',
        transformStyle: 'preserve-3d',
        backfaceVisibility: 'hidden',
        transform: 'translateZ(0)',}}>
        <div className='pointer-events-none'>
          <h1 className='text-4xl font-bold text-center pointer-events-none'
          >The Music Pit</h1>

          {prediction !== null ?
          <p className='text-center pointer-events-none'
          >Predicted Enjoyment: {prediction}/10</p> : <p>Loading...</p>}

          {selectedSong && <p className='text-center pointer-events-none'
          >{selectedSong.name} by {selectedSong.artists}</p>}
        </div>
      </Html>
    </>
  )
}

const InputField = ({
    songs,
    selectedSong,
    searchQuery,
    setSearchQuery,
    setSelectedSong,
    setImgUrl,
  }: {
    prediction:number|null,
    songs:song[],
    selectedSong:song|null,
    searchQuery:string|null,
    setSearchQuery: (searchQueary:string) => void,
    setSelectedSong: (selectedSong:song) => void,
    setImgUrl: (imgUrl:string) => void
  }) => {

  const [showDropdown, setShowDropdown] = useState<boolean>(false)

  return (
    <div className='absolute w-[80%] z-1 left-[10%] top-[45%]'
    >
      <input // The input box
        type='text'
        value={searchQuery||''}
        onChange={ (e)=>{
          setSearchQuery(e.currentTarget.value)
          if(e.currentTarget.value) setShowDropdown(true)
          else setShowDropdown(false)
        }}
        className="text-white bg-gray-800 w-[100%]"
        // After clicking away the dropdown dissappears after a delay
        onBlur={() => setTimeout(() => setShowDropdown(false), 200)} // Give time for click away to register
        placeholder='Search For Ya Tunes'
        />

      {/*Only display the dropdown if we're supposed to, and ther's results to show*/}
        <ul className={`transition-all duration-300 bg-black border-[1px] border-solid border-[#ccc] m-0 p-0 w-[100%] overflow-y-auto ${(showDropdown && songs.length > 0) ? 'pointer-events-auto opacity-100 max-h-[300px]' : 'pointer-events-auto opacity-0 max-h-0'}`}>
          {songs.map(song => (
            <li // Each of these is a song in the list
              className='p-[0.5rem] cursor-pointer'
              key={song.id}
              style={{padding: '0.5rem', cursor: 'pointer'}}
              onClick={() => {
                if(selectedSong?.id !== song.id) {
                  setImgUrl(song.images)
                  setSelectedSong(song)
                  
                }
              }}
              >
                {song.name} - {song.artists[0]}
            </li>
          ))}
        </ul>
  </div>
  )
}

function App() {

  // Holds the search results
  const [songs, setSearchResults] = useState<song[]>([{
    album: '',
    artists: [''],
    id: '',
    images: '',
    name: ''
  }])

  const [sidebarOpen, setSidebarOpen] = useState(false)

  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  const [imgUrl, setImgUrl] = useState<string>(null!)
  const [selectedSong, setSelectedSong] = useState<song>()
  const [prediction, setPrediction] = useState<number | null>(null)
  const [searchQuery, setSearchQuery] = useState<string>('')

  // Fetch the prediction once after the pages mounts
  useEffect(() => {
    fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json'},
      body: JSON.stringify({ features: [0.6, 0.8, 0.3] }) // Fake data
    })
    .then(res => res.json())
    .then(data => {
      setPrediction(data.predicted_enjoyment)
    })

    document.addEventListener('mousemove', (event) => {
      setMousePosition({x:event.pageX, y:event.pageY})
    })
  }, [])

  useEffect(() => {
    // Return if there's no search term
    if(!searchQuery.length){return}

    // Every time a key is pressed, start the timer and display new results if it's been 300ms
    const timeout = setTimeout(() => {
      fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({query: searchQuery})
      })
      .then(response => response.json())
      .then(items => setSearchResults(items))
    }, 300) // Only fetch after a delay of no typing

    return () => clearTimeout(timeout)
  }, [searchQuery])
  
  // Visual and interactive elements
  return (
      <div className='app-container'>

        {/*Creates a router*/}
        <BrowserRouter>
          {/*Holds the routes*/}
          <Routes>
            {/* A route to the callback React element for when spotify calls back localhost:5173 */}
            <Route path={'/auth/callback'} element={<AuthCallback />} />
            <Route path="/spotifind" element={<></>} />
            <Route path="*" element={<RedirectHandler />} />
            <Route path="" element={<RedirectHandler />} />
          </Routes>
        </BrowserRouter>


        <div className='canvas-container'>
          <Canvas className='interacting-canvas'>

            <ambientLight intensity={Math.PI / 2}/>

            {/* Makes sure the album only displays if there's a song selected*/}
            {selectedSong?.id && <Album key={selectedSong.id} imgUrl={imgUrl}/>}
          </Canvas>
        </div>
        
        <div className='relative w-[40vw] h-[100vh] bg-[#222222] flex flex-col items-center flex-start text-white'>
          
          <div className={`h-screen ${sidebarOpen ? 'w-fit' : 'w-50'} absolute self-start col-start-1 z-50`}>
            <Sidebar isOpen={sidebarOpen} toggle={()=>{setSidebarOpen(!sidebarOpen)}} />
          </div>

          <div className='absolute h-[30%] w-[100%] top-[10%] col-start-1 z-0'>
            <Canvas className='z-0'>
              <SearchUI3D
              prediction={prediction}
              selectedSong={selectedSong||null}
              mousePosition={mousePosition}
              ></SearchUI3D>
            </Canvas>
          </div>

          <InputField
            prediction={prediction}
            songs={songs}
            selectedSong={selectedSong||null}
            searchQuery={searchQuery}
            setSearchQuery={setSearchQuery}
            setSelectedSong={setSelectedSong}
            setImgUrl={setImgUrl}
          />
        </div>
      </div>
  )
}

export default App
