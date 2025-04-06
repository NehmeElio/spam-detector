import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the Spam or Ham Detector heading', () => {
  render(<App />);
  const headingElement = screen.getByText(/spam or ham detector/i);  // Checks if the heading text is rendered
  expect(headingElement).toBeInTheDocument();  // Asserts that the heading is in the document
});
