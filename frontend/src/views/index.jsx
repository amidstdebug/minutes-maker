// src/App.jsx

import React, { useState } from 'react';
import { Container, Flex, Box, Heading, Button, useColorModeValue, useColorMode } from '@chakra-ui/react';
import { FileUploader } from '../components/FileUpload'; // Adjust the import path as necessary
import { TranscriptionDisplay } from '../components/TranscriptionDisplay'; // Adjust the import path as necessary
import { MinutesDisplay } from '../components/MinutesDisplay'; // Adjust the import path as necessary

function App() {
  const [transcription, setTranscription] = useState('');
  const { colorMode, toggleColorMode } = useColorMode();

  const handleTranscription = (newTranscription) => {
    setTranscription(newTranscription);
  };

  return (
    <Container maxW="container.xl" p={5} centerContent>
      <Heading mb={6}>Audio Transcription App</Heading>
      <Button colorScheme='blue' onClick={toggleColorMode} mb={6}>
        Switch to {colorMode === 'light' ? 'Dark' : 'Light'} Mode
      </Button>
      <Flex direction={{ base: "column", md: "row" }} gap={6} align="flex-start">
        <Box flex="0.3" minWidth="300px"> {/* Adjusted for 30% space */}
          <FileUploader onTranscription={handleTranscription} />
        </Box>
        <Flex direction="column" flex="0.7" minWidth="700px" gap={4}> {/* Adjusted for 70% space */}
          <TranscriptionDisplay transcription={transcription} />
          <MinutesDisplay transcriptionLength={transcription.length} />
        </Flex>
      </Flex>
    </Container>
  );
}

export default App;
