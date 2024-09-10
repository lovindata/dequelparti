interface Props extends React.InputHTMLAttributes<HTMLInputElement> {
  title: string;
  subTitle: string;
  children?: React.ReactNode;
}

export function Section({ title, subTitle, children, className }: Props) {
  return (
    <section className="flex flex-col items-center space-y-5 md:space-y-11">
      <header className="flex flex-col items-center space-y-3 md:space-y-7">
        <h1 className="text-center text-3xl font-semibold md:text-5xl">
          {title}
        </h1>
        <p className="text-center text-base font-semibold text-muted-foreground md:text-3xl">
          {subTitle}
        </p>
      </header>
      <div className={className}>{children}</div>
    </section>
  );
}
