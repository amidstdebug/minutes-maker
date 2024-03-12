// src/components/TranscriptionDisplay.jsx

import React from 'react';
import { Box, Text } from '@chakra-ui/react';

export const TranscriptionDisplay = ({ transcription }) => {
  return (
    <Box height="200px" overflowY="auto" p={4} border="1px" borderColor="gray.200" borderRadius="md">
      <Text>{transcription || "No transcription yet..."}</Text>
    </Box>
  );
};
