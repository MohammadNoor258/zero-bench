import { render, screen, fireEvent } from '@testing-library/react';
import Counter from '@/app/Counter';

describe('Counter', () => {
  it('increments the counter', () => {
    render(<Counter />);

    const button = screen.getByRole('button', { name: 'Count: 0' });

    fireEvent.click(button);

    expect(screen.getByRole('button', { name: 'Count: 1' })).toBeInTheDocument();
  });
});
