// src/components/MinutesDisplay.jsx

import React from 'react';
import { Box, Text } from '@chakra-ui/react';

export const MinutesDisplay = ({ transcriptionLength }) => {
  // Convert length to minutes or however you're calculating it
  return (
    <Box height="200px" overflowY="auto" p={4} border="1px" borderColor="gray.200" borderRadius="md">
      <Text>No minutes generated yet...</Text>
    </Box>
  );
};
