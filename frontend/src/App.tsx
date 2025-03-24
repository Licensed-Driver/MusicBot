import { useEffect, useState, useRef } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
//import html2canvas from 'html2canvas' // Part of the HTML to canvas and plane stuff that is useful but not being used
import'./App.css'

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

type song = {
  album: string,
  artists: string[],
  id: string,
  images: string,
  name: string
}

function Album({imgUrl, ...props}:any) {

  // Loader for loading the album textures
  const loader = new THREE.TextureLoader()
    
  // Set up state for hovering and active states
  const [hovered, setHover] = useState(false)
  const [active, setActive] = useState(false)

  // Get mouse position
  const { mouse } = useThree()

  // Using a reference gives direct access to the mesh
  const meshRef = useRef<THREE.Mesh>(null!)
  // Lets us update the texture since it needs the texture to be constant and local
  const [albumTex, setTexture] = useState<THREE.Texture | null>(null)

  useEffect(() => {

    // Load the new texture every time the url is changed
    loader.load(imgUrl, (loadedTexture) => {
      setTexture(loadedTexture)
      const material = meshRef.current.material as THREE.MeshBasicMaterial
      material.needsUpdate = true
    })

    console.log(imgUrl)
  }, [imgUrl])

  const [[positionX, positionY, positionZ], setPosition] = useState([0, 0, 0])
  const [scale, setScale] = useState({x:5, y:5, z:5})

  const timer = useRef(0)

  useFrame((_, delta) => {
    // In case the mesh isn't set yet
    if(!meshRef.current) return

    // Get the x and y scaled down a lil
    const targetX = mouse.y * -0.3
    const targetY = mouse.x * 0.3

    // Linearly interpolate towards the mouse (smooth movement just like your brain)
    meshRef.current.rotation.x = THREE.MathUtils.lerp(meshRef.current.rotation.x, targetX, 0.1)
    meshRef.current.rotation.y = THREE.MathUtils.lerp(meshRef.current.rotation.y, targetY, 0.1)

    timer.current += delta

    // If the album is hovered over it'll expand over time for a smooth interaction
    if(hovered) setScale({x:THREE.MathUtils.lerp(meshRef.current.scale.x, 6.5, 0.1), y:THREE.MathUtils.lerp(meshRef.current.scale.y, 6.5, 0.1), z:1})
    else if ((scale.x !== meshRef?.current.scale.x) || (scale.y !== meshRef?.current.scale.y)) setScale({x:THREE.MathUtils.lerp(meshRef.current.scale.x, 6.5, 0.1), y:THREE.MathUtils.lerp(meshRef.current.scale.y, 6.5, 0.1), z:1})

    setPosition([0, Math.sin(timer.current), 0])
  })

  return (
    <mesh
      {...props} // The properties passed automaticall
      ref={meshRef} // A reference to the object directly
      scale={scale} // Scales it up if it's active
      onClick={() => setActive(!active)} // Changes it's active state if it's clicked
      onPointerOver={() => setHover(true)} // This and one below track hovering
      onPointerOut={() => setHover(false)}
      position={[positionX, positionY, positionZ]}>
      <planeGeometry args={[5, 5, 1]}/>
      <meshBasicMaterial map={albumTex}/>
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
  setScreenPos,
}: {
  prediction:number|null,
  selectedSong:song|null,
  setScreenPos: ({x, y}:{x:number, y:number}) => void
}) {

  const { camera, size } = useThree();

  const time = useRef(0)

  useFrame((_, delta) => {
    // Store the position vector
    const vector = new THREE.Vector3();
    // Get the vector from world position matrix
    vector.setFromMatrixPosition(groupRef.current.matrixWorld);
    // Project to da camera so we know where you are on the screen
    vector.project(camera);
    // Get the x and y positions on the screen with some mathemagic
    const x = (vector.x * 0.5 + 0.5) * size.width;
    const y = (-vector.y * 0.5 + 0.5) * size.height;

    setScreenPos({ x, y });

    time.current += delta * 2

    setRotation([rotationX, Math.sin(time.current)/2])
  });

  const [[rotationX, rotationY], setRotation] = useState<number[]>([0, 0])
  
  // For the world position of the 3D elements
  const groupRef = useRef<THREE.Group>(null!)
  // For the reference to the actual Html drei element
  const htmlRef = useRef<HTMLDivElement>(null!)

  return (
    <>
    <group ref={groupRef} position={[0, 0, 0]}>
      <Html pointerEvents='auto' ref={htmlRef} scale={[0.8, 0.8, 0.5]} position={[0,0,0]} rotation={[rotationX, rotationY, 0]} transform>
        <div className='SearchUI3D'>
          <h1>The Music Pit</h1>

          {prediction !== null ? <p>Predicted Enjoyment: {prediction}/10</p> : <p>Loading...</p>}

          {selectedSong && <p>{selectedSong.name} by {selectedSong.artists}</p>}
        </div>
      </Html>
    </group>
    </>
  )
}

const ProjectedInput = ({
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
    screenPos:{x:number, y:number}
    setSearchQuery: (searchQueary:string) => void,
    setSelectedSong: (selectedSong:song) => void,
    setImgUrl: (imgUrl:string) => void
  }) => {
  
  const handleKeyDown = async (e:any) => {
    e.preventDefault()

    if(e.key.length===1) {
      // Just regular keys
      setSearchQuery(searchQuery + e.key)
    } else if(e.key.slice(0, 4)==='Shift') {
      setSearchQuery(searchQuery + e.key.slice(0,1).toUpperCase())
    } else if(e.key === 'Backspace') {
      setSearchQuery(searchQuery?.slice(0, -1)||'')
    } else if(e.key==='Enter') {
      console.log("Search:", searchQuery)
    }

    if(searchQuery) setShowDropdown(true)
  }

  const [showDropdown, setShowDropdown] = useState<boolean>(false)

  return (
    <div className='search-bar'
      style={{}}
    >
      <input // The input box
        type='text'
        value={searchQuery||''}
        onKeyDown={handleKeyDown}
        // After clicking away the dropdown dissappears after a delay
        onBlur={() => setTimeout(() => setShowDropdown(false), 200)} // Give time for click away to register
        placeholder='Search For Ya Tunes'
        />

      {showDropdown && songs.length > 0 && ( // Only display the dropdown if we're supposed to, and ther's results to show
        <ul className="dropdown">
          {songs.map(song => (
            <li // Each of these is a song in the list
              className='dropdown li'
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
      )}
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

  const [screenPos, setScreenPos] = useState({ x: 0, y: 0 });

  const [imgUrl, setImgUrl] = useState<string>(null!)
  const [selectedSong, setSelectedSong] = useState<song>()
  const [prediction, setPrediction] = useState<number | null>(null)
  const [searchQuery, setSearchQuery] = useState<string>('')

  // Fetch the prediction once after the pages mounts
  useEffect(() => {
    fetch('https://d448-70-74-152-126.ngrok-free.app/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json'},
      body: JSON.stringify({ features: [0.6, 0.8, 0.3] }) // Fake data
    })
    .then(res => res.json())
    .then(data => {
      setPrediction(data.predicted_enjoyment)
    })
  }, [])

  useEffect(() => {
    // Return if there's no search term
    if(!searchQuery.length){return}

    // Every time a key is pressed, start the timer and display new results if it's been 300ms
    const timeout = setTimeout(() => {
      fetch('https://d448-70-74-152-126.ngrok-free.app/search', {
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

        <div className='canvas-container'>
          <Canvas className='interacting-canvas'>

            <ambientLight intensity={Math.PI / 2}/>

            {/* Makes sure the album only displays if there's a song selected*/}
            {selectedSong?.id && <Album key={selectedSong.id} imgUrl={imgUrl}/>}
          </Canvas>
        </div>
        
        <div className="search-ui-container">
          <div className="search-canvas-container">
            <Canvas>
              <SearchUI3D
              prediction={prediction}
              selectedSong={selectedSong||null}
              setScreenPos={setScreenPos}
              ></SearchUI3D>
            </Canvas>
          </div>

          <ProjectedInput
            prediction={prediction}
            songs={songs}
            selectedSong={selectedSong||null}
            searchQuery={searchQuery}
            screenPos={screenPos}
            setSearchQuery={setSearchQuery}
            setSelectedSong={setSelectedSong}
            setImgUrl={setImgUrl}
          />
        </div>
      </div>
  )
}

export default App
