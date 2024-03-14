// src/components/FileUpload.jsx

import React, { useState, useRef } from 'react';
import { Button, Input, VStack, Box, Text } from '@chakra-ui/react';

export const FileUploader = ({ onTranscription }) => {
  const [fileName, setFileName] = useState('');
  const [fileURL, setFileURL] = useState(null); // State to store the audio file URL
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setFileName(file.name);
      const url = URL.createObjectURL(file); // Create a URL for the uploaded file
      setFileURL(url); // Set the file URL to be used in the audio player

      // Simulate sending the file to a server and receiving a transcription response
      setTimeout(() => {
        onTranscription(`Transcription of ${file.name}`);
      }, 1500);
    }
  };

  return (
    <Box
      maxW="sm"
      borderWidth="1px"
      borderRadius="lg"
      overflow="hidden"
      padding={4}
      boxShadow="lg"
      bg="white"
    >
      <VStack spacing={4}>
        <Input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          hidden
          accept="audio/*" // Ensure only audio files can be uploaded
        />

        {fileName && (
          <Box p={2} bg="gray.100" borderRadius="md" width="full">
            <Text fontSize="sm" wordBreak="break-word">
              Uploaded File: {fileName}
            </Text>
          </Box>
        )}
        {fileURL && (
          <Box p={2} display="flex" justifyContent="center" width="full"> {/* Updated for centering */}
            <audio controls src={fileURL}>
              Your browser does not support the audio element.
            </audio>
          </Box>
        )}
        <Button
          colorScheme="blue"
          onClick={() => fileInputRef.current.click()}
        >
          Upload Audio File
        </Button>
      </VStack>
    </Box>
  );
};
