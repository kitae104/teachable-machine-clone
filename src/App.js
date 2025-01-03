import React, { useEffect, useRef, useState } from 'react';
import Card from './components/cards';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
	faPencilAlt,
	faVideo,
	faTrash,
} from '@fortawesome/free-solid-svg-icons';
import * as tf from '@tensorflow/tfjs';
import { shuffle } from '@tensorflow/tfjs-core/dist/util';
import HorizontalBarChart from './components/HorizontalBarChart';

const savedImages = {};

/* tf.setBackend('tensorflow'); 
await tf.ready();
console.log('TensorFlow.js is using CPU backend'); */

function App() {
	let [labels, setLabels] = useState([
		{ id: 1, name: '라벨 1', totalImages: 0 },
		{ id: 2, name: '라벨 2', totalImages: 0 },
	]);
	const [loadingProgress, setLoadingProgress] = useState(0);
	const [mobilenetReady, setMobilenetReady] = useState(false);
	const [mobilenet, setMobilenet] = useState(null);
	const [model, setModel] = useState(null);
	const [trainingDataInputs, setTrainingDataInputs] = useState([]);
	const [trainingDataOutputs, setTrainingDataOutputs] = useState([]);
	const [isTraining, setIsTraining] = useState(false);
	const [isPredict, setIsPredict] = useState(false);
	const [epochsCompleted, setEpochsCompleted] = useState(0);
	const [trainingResults, setTrainingResults] = useState(false);
	const [finishedTraining, setFinishedTraining] = useState(false);
	const [predictionArray, setPredictionArray] = useState([0, 0, 0]);
	const totalEpochs = 25;
	const fileInputRef = useRef(
		Array(labels.length)
			.fill()
			.map(() => React.createRef())
	);
	const fileInputPredict = useRef(null);
	const [webcamEnabled, setWebcamEnabled] = useState(false);
	const [predictImageURL, setPredictImageURL] = useState(null);
	const [showModal, setShowModal] = useState(true);

	async function loadMobileNetModel() {
		try {
			const URL =
				'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
			const mobilenet = await tf.loadGraphModel(URL, {
				fromTFHub: true,
				onProgress: (fraction) => setLoadingProgress(fraction * 100),
			});
			mobilenet.predict(tf.zeros([1, 224, 224, 3]));
			return mobilenet;
		} catch (error) {
			console.error('Error loading MobileNet model:', error);
			return null;
		}
	}

	async function loadTfModel() {
		const model = await tf.sequential();
		return model;
	}

	useEffect(() => {
		loadMobileNetModel().then((mobilenet) => {
			if (mobilenet) {
				setMobilenet(mobilenet);
				setMobilenetReady(true);
			}
		});
		loadTfModel().then((model) => {
			if (model) {
				setModel(model);
			}
		});
		if (mobilenetReady) {
			setShowModal(false);
		}
	}, [mobilenetReady]);

	const videoRef = useRef(null);
	const canvasRef = useRef(null);
	const [savedImages, setSavedImages] = useState([]); // State to store captured images

	const handleAddLabel = () => {
		const newId = labels.length + 1;
		setLabels([
			...labels,
			{ id: newId, name: `라벨 ${newId}`, totalImages: 0 },
		]);
		setSavedImages([...savedImages, []]);
	};

	const handleUploadImage = async (event, index) => {
		const file = event.target.files[0];
		const canvas = canvasRef.current;
		const context = canvas.getContext('2d');
		const image = new Image();
		const loadImage = new Promise((resolve, reject) => {
			image.onload = () => resolve();
			image.onerror = (error) => reject(error);
		});

		image.src = URL.createObjectURL(file);
		try {
			await loadImage;
			context.clearRect(0, 0, canvas.width, canvas.height);
			context.drawImage(image, 0, 0, canvas.width, canvas.height);
			const imageDataURL = canvas.toDataURL();
			const imageData = context.getImageData(
				0,
				0,
				canvas.width,
				canvas.height
			);
			imageProcessing(imageData, imageDataURL, index);
		} catch (error) {
			console.error('이미지 로딩 오류:', error);
		}
	};

	const handleOpenFilePicker = (index) => {
		fileInputRef.current[index].click();
	};

	const handleImageCapture = async (index) => {
		if (!webcamEnabled) return alert('먼저 카메라를 켜세요');
		const video = videoRef.current;
		const canvas = canvasRef.current;
		const context = canvas.getContext('2d');

		context.drawImage(video, 0, 0, canvas.width, canvas.height);

		const imageDataURL = canvas.toDataURL();
		const imageData = context.getImageData(0, 0, 224, 224);
		imageProcessing(imageData, imageDataURL, index);
	};

	const handleDeleteLabel = (index) => {
		const newLabels = [...labels];
		newLabels.splice(index, 1);
		setLabels(newLabels);
		const newSavedImages = [...savedImages];
		newSavedImages.splice(index, 1);
		setSavedImages(newSavedImages);
	};

	const imageProcessing = async (imageData, imageDataURL, index) => {
		const features = await calculateFeaturesFromImageData(imageData);
		console.log(typeof imageDataURL);
		console.log(imageData, imageDataURL, index, features);
		setTrainingDataInputs((prevInputs) => [...prevInputs, features]);
		setTrainingDataOutputs((prevOutputs) => [...prevOutputs, index]);
		const newSavedImages = [...savedImages];
		if (!newSavedImages[index]) {
			newSavedImages[index] = [];
		}
		newSavedImages[index].push(imageDataURL);
		setSavedImages(newSavedImages);
		const newLabels = [...labels];
		newLabels[index].totalImages += 1;
		setLabels(newLabels);
	};

	async function trainModel() {
		setEpochsCompleted(0);

		const filteredLabels = labels.filter(
			(label, index) => savedImages[index]?.length > 0
		);

		setIsTraining(true);
		// Prepare training data for the filtered labels
		const filteredInputs = [];
		const filteredOutputs = [];

		filteredLabels.forEach((label, index) => {
			filteredInputs.push(trainingDataInputs[index]);
			filteredOutputs.push(trainingDataOutputs[index]);
		});

		shuffle(trainingDataInputs, trainingDataOutputs);
		let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
		let oneHotOutputs = tf.oneHot(outputsAsTensor, labels.length);
		let inputsAsTensor = tf.stack(trainingDataInputs);
		model.add(
			tf.layers.dense({
				inputShape: [1024],
				units: 128,
				activation: 'relu',
			})
		);
		model.add(
			tf.layers.dense({ units: labels.length, activation: 'softmax' })
		);

		model.summary();

		model.compile({
			optimizer: 'adam',
			loss:
				labels.length === 2
					? 'binaryCrossentropy'
					: 'categoricalCrossentropy',
			metrics: ['accuracy'],
		});

		let results = await model.fit(inputsAsTensor, oneHotOutputs, {
			shuffle: true,
			batchSize: 5,
			epochs: totalEpochs,
			callbacks: {
				onEpochEnd: logProgress,
			},
		});

		function logProgress(epoch, logs) {
			setEpochsCompleted((prevEpochs) => prevEpochs + 1);
			console.log('Data for epoch ' + epoch, logs);
		}

		outputsAsTensor.dispose();
		oneHotOutputs.dispose();
		inputsAsTensor.dispose();
		setTrainingResults(results);
		setIsTraining(false);
		setFinishedTraining(true);
	}

	const handleDeleteImage = (labelIndex, imageIndex) => {
		const newSavedImages = [...savedImages];
		newSavedImages[labelIndex].splice(imageIndex, 1); // Hapus gambar dari daftar
		setSavedImages(newSavedImages);
		const newLabels = [...labels];
		newLabels[labelIndex].totalImages -= 1;
		setLabels(newLabels);
		setTrainingDataInputs(
			trainingDataInputs.filter(
				(_, index) => trainingDataOutputs[index] !== labelIndex
			)
		);
		setTrainingDataOutputs(
			trainingDataOutputs.filter((output) => output !== labelIndex)
		);
	};

	async function calculateFeaturesFromImageData(imageData) {
		const tensor = tf.browser.fromPixels(imageData);
		const resizedTensor = tf.image.resizeBilinear(tensor, [224, 224], true);
		const normalizedTensor = resizedTensor.div(255);
		const predictions = mobilenet.predict(normalizedTensor.expandDims());

		const features = predictions.squeeze();
		tensor.dispose();
		normalizedTensor.dispose();
		resizedTensor.dispose();
		predictions.dispose();
		return features;
	}

	function calculateFeaturesOnCurrentFrame() {
		const video = videoRef.current;
		const tensor = tf.browser.fromPixels(video);
		const resizedTensor = tf.image.resizeBilinear(tensor, [224, 224], true);
		const normalizedTensor = resizedTensor.div(255);
		const predictions = mobilenet.predict(resizedTensor.expandDims());

		const features = predictions.squeeze();
		tensor.dispose();
		normalizedTensor.dispose();
		resizedTensor.dispose();
		predictions.dispose();
		return features;
	}

	const handleUploadPredictImage = async (event) => {
		const file = event.target.files[0];
		const canvas = canvasRef.current;
		const context = canvas.getContext('2d');
		const image = new Image();
		const loadImage = new Promise((resolve, reject) => {
			image.onload = () => resolve();
			image.onerror = (error) => reject(error);
		});

		image.src = URL.createObjectURL(file);
		try {
			await loadImage;
			context.clearRect(0, 0, canvas.width, canvas.height);
			context.drawImage(image, 0, 0, canvas.width, canvas.height);
			const imageDataURL = canvas.toDataURL();
			setPredictImageURL(imageDataURL);
			const imageData = context.getImageData(
				0,
				0,
				canvas.width,
				canvas.height
			);
			predictWithImage(imageData);
		} catch (error) {
			console.error('이미지 로딩 오류 :', error);
		}
	};

	async function predictWithImage(imageData) {
		let features = await calculateFeaturesFromImageData(imageData);
		let prediction = model.predict(features.expandDims());
		let predictionArray = prediction.arraySync();
		setIsPredict(true);
		setPredictionArray(predictionArray);
	}

	async function predictLoop() {
		setPredictImageURL(null);
		while (true && webcamEnabled) {
			await new Promise((resolve) => setTimeout(resolve, 1000));
			let imageFeatures = await calculateFeaturesOnCurrentFrame();
			let prediction = model.predict(imageFeatures.expandDims());
			let predictionArray = prediction.arraySync();
			console.log(predictionArray);
			setIsPredict(true);
			setPredictionArray(predictionArray);
		}
	}

	const handleEnableWebcam = () => {
		navigator.mediaDevices
			.getUserMedia({ video: true })
			.then((stream) => {
				videoRef.current.srcObject = stream;
				setWebcamEnabled(true);
			})
			.catch((error) => {
				console.error('웹캠 접근 오류 :', error);
			});
	};

	const handleDisableWebcam = () => {
		const video = videoRef.current;
		if (video && video.srcObject) {
			const stream = video.srcObject;
			const tracks = stream.getTracks();
			tracks.forEach((track) => track.stop());
			video.srcObject = null;
			setWebcamEnabled(false);
		}
	};

	const handleLabelChange = (index, newName) => {
		if (!newName) return alert('라벨 이름은 비워둘 수 없습니다.');
		const newLabels = [...labels];
		newLabels[index].name = newName;
		setLabels(newLabels);
	};

	return (
		<div>
			{showModal && (
				<div className="fixed top-0 left-0 w-full h-full bg-black bg-opacity-50 flex justify-center items-center">
					<div className="bg-white p-4 rounded shadow-md">
						<div className="flex justify-between mb-1">
							{' '}
							잠시만 기다려주세요. 애플리케이션이 시작됩니다.
						</div>
						<div className="w-full bg-slate-200 rounded-full h-2.5">
							<div
								className="bg-blue-600 h-2.5 rounded-full"
								style={{ width: `${loadingProgress}%` }}
							></div>
						</div>
					</div>
				</div>
			)}
			<div className="py-6 text-center bg-blue-500">
				<h1 className="text-4xl font-bold text-white">
					Teachable Machine
				</h1>
			</div>
			{webcamEnabled && (
				<div className="border border-gray-300 px-4 py-2 mb-2 content-center">
					<video
						ref={videoRef}
						className="mx-auto"
						width="640"
						height="480"
						autoPlay
						muted
					></video>
				</div>
			)}
			<div className="container my-4 mx-auto flex flex-col md:flex-row gap-8">
				<div className="border border-gray-300 rounded-lg rounded-lg basis-1/2 shadow-md p-4">
					<div className="overflow-x-auto">
						<div className="grid grid-cols-1 gap-4">
							{labels.map((label, index) => (
								<div
									key={index}
									className="border border-gray-300 rounded-lg px-4 py-2 mb-2 flex flex-col"
								>
									<div className="flex justify-between items-center mb-2">
										<span
											contentEditable
											suppressContentEditableWarning
											onBlur={(e) =>
												handleLabelChange(
													index,
													e.target.textContent
												)
											}
											className="max-w-sm overflow-hidden font-bold whitespace-nowrap pr-8"
										>
											{label.name}
										</span>
										<FontAwesomeIcon
											icon={faTrash}
											className={`top-0 right-0 text-red-500 cursor-pointer ${
												isPredict ? 'hidden' : ''
											}`}
											onClick={() =>
												handleDeleteLabel(index)
											}
										/>
									</div>

									<div className="border border-gray-300 px-4 py-2 mb-2 overflow-x-auto flex ">
										<canvas
											ref={canvasRef}
											width="224"
											height="224"
											hidden
										></canvas>

										{savedImages[index] &&
											savedImages[index].map(
												(image, i) => (
													<div
														key={i}
														className="mr-2"
													>
														<div className="overflow-x-auto flex mr-2 relative ">
															<img
																src={image}
																alt={`Captured Image 1 - ${i}`}
																style={{
																	maxWidth:
																		'80px',
																	maxHeight:
																		'60px',
																}}
															/>
															<FontAwesomeIcon
																icon={faTrash}
																className={`top-0 right-0 text-red-500 cursor-pointer ${
																	isPredict
																		? 'hidden'
																		: ''
																}`}
																onClick={() =>
																	handleDeleteImage(
																		index,
																		i
																	)
																}
															/>
														</div>
													</div>
												)
											)}
									</div>
									<div className="row">
										<span className="text-gray-500 text-sm">
											총 이미지: {label.totalImages}
										</span>
									</div>
									<div className="flex flex-col md:flex-row gap-8 justify-center">
										<input
											type="file"
											accept="image/*"
											ref={(element) =>
												(fileInputRef.current[index] =
													element)
											}
											style={{ display: 'none' }}
											onChange={(event) =>
												handleUploadImage(event, index)
											}
										/>
										<button
											onClick={() =>
												handleOpenFilePicker(index)
											}
											className={`flex-1/2 bg-blue-600 ${
												isPredict
													? ''
													: 'hover:bg-blue-700'
											}  hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline`}
											disabled={isPredict}
										>
											이미지
										</button>
										<button
											onClick={() =>
												handleImageCapture(index)
											}
											className={`flex-2 bg-blue-600 ${
												isPredict
													? ''
													: 'hover:bg-blue-700'
											}  hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline`}
											disabled={isPredict}
										>
											웹 캠
										</button>
									</div>
								</div>
							))}
						</div>
						<div className="grid grid-cols-1 gap-4">
							<button
								onClick={handleAddLabel}
								className="border border-dashed border-gray-300 px-4 py-6 mb-2 flex flex-col"
								disabled={isPredict}
							>
								<span className="text-gray-500 mx-auto my-auto">
									라벨 추가
								</span>
							</button>
						</div>
					</div>
				</div>

				<div className="bg-white rounded-lg basis-1/4 shadow-md p-4">
					<div className="col-md-12 flex flex-col gap-8 justify-center mt-4">
						{webcamEnabled ? (
							<button
								onClick={handleDisableWebcam}
								className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded"
							>
								카메라 끄기{' '}
								<FontAwesomeIcon
									icon={faVideo}
									className="ml-2"
								/>
							</button>
						) : (
							<button
								onClick={handleEnableWebcam}
								className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded"
							>
								카메라 켜기{' '}
								<FontAwesomeIcon
									icon={faVideo}
									className="ml-2"
								/>
							</button>
						)}
						<button
							onClick={() => {
								trainModel();
							}}
							className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
							disabled={
								finishedTraining
									? true
									: isTraining
									? true
									: false
							}
						>
							{isTraining &&
								'처리중 ' +
									epochsCompleted +
									'/' +
									totalEpochs * labels.length}
							{!isTraining && finishedTraining && '처리 완료'}
							{!isTraining &&
								!finishedTraining &&
								'프로세스 모델'}
						</button>
						<input
							type="file"
							accept="image/*"
							ref={(element) =>
								(fileInputPredict.current = element)
							}
							style={{ display: 'none' }}
							onChange={(event) =>
								handleUploadPredictImage(event)
							}
						/>
						<button
							onClick={() =>
								webcamEnabled
									? alert(
											'이 기능을 사용하려면 카메라를 꺼주세요'
									  )
									: fileInputPredict.current.click()
							}
							className={`${
								finishedTraining
									? 'bg-blue-600 hover:bg-blue-700'
									: 'bg-red-700'
							}  text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline`}
							disabled={finishedTraining ? false : true}
						>
							이미지를 이용한 예측
						</button>
						<button
							onClick={() =>
								!webcamEnabled
									? alert(
											'이 기능을 사용하려면 카메라를 켜세요.'
									  )
									: predictLoop()
							}
							className={`${
								finishedTraining
									? 'bg-blue-600 hover:bg-blue-700'
									: 'bg-red-700'
							}  text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline`}
							disabled={finishedTraining ? false : true}
						>
							카메라를 이용한 예측
						</button>
					</div>
					{isTraining && (
						<div className="loading-bar">
							잠시 기다려 주세요. 처리 중입니다. ({epochsCompleted}
							/{totalEpochs * labels.length})
						</div>
					)}
				</div>

				<div className="bg-white rounded-lg shadow-md basis-1/4 p-4">
					{isPredict && predictImageURL && (
						<img
							src={predictImageURL}
							style={{ maxWidth: '224px', maxHeight: '224px' }}
						/>
					)}
					<div className="flex justify-between items-center mb-2">
						<h2 className="text-xl font-medium">예측 결과</h2>
					</div>
					{isPredict && (
						<HorizontalBarChart
							predictionArray={predictionArray[0]}
							labels={labels}
						/>
					)}
				</div>
			</div>
		</div>
	);
}

export default App;
