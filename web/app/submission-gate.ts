// Acquire before the first await; React state alone cannot guard the same tick.
export function createSubmissionGate() {
  let occupied = false;
  return {
    acquire(): (() => void) | null {
      if (occupied) return null;
      occupied = true;
      let released = false;
      return () => {
        if (released) return;
        released = true;
        occupied = false;
      };
    },
  };
}
