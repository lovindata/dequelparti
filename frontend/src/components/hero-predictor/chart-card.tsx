import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/src/components/shadcn/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/src/components/shadcn/ui/chart";
import { Activity } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  XAxis,
  YAxis,
} from "recharts";

interface Props {
  chartData: { [key: string]: number } | undefined;
}

export function ChartCard({ chartData }: Props) {
  let chartDataDisplayable: { affiliation: string; percent: number }[] = [];
  if (chartData !== undefined) {
    const sumValue = Object.values(chartData).reduce((a, b) => a + b, 0);
    chartDataDisplayable = Object.entries(chartData).map(
      ([affiliation, value]) => ({
        affiliation,
        percent: sumValue === 0 ? 0 : Math.round((value / sumValue) * 100),
      }),
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Affiliation Politique</CardTitle>
        <CardDescription>En pourcentage</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer
          config={{
            percent: {
              label: "Pourcentage",
              color: "hsl(var(--chart-1))",
            },
          }}
        >
          <BarChart
            accessibilityLayer
            data={chartDataDisplayable}
            layout="vertical"
            margin={{
              right: 16,
            }}
          >
            <CartesianGrid horizontal={false} />
            <YAxis
              dataKey="affiliation"
              type="category"
              tickLine={false}
              tickMargin={10}
              axisLine={false}
              tickFormatter={(value) => value.slice(0, 3)}
              hide
            />
            <XAxis dataKey="percent" type="number" hide />
            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent indicator="line" />}
            />
            <Bar
              dataKey="percent"
              layout="vertical"
              fill="var(--color-percent)"
              radius={4}
            >
              <LabelList
                dataKey="affiliation"
                position="insideLeft"
                offset={8}
                className="fill-[--color-label]"
                fontSize={12}
              />
              <LabelList
                dataKey="percent"
                position="right"
                offset={8}
                className="fill-foreground"
                fontSize={12}
              />
            </Bar>
          </BarChart>
        </ChartContainer>
      </CardContent>
      <CardFooter className="flex-col items-start gap-2 text-sm">
        <div className="flex animate-pulse gap-2 font-medium leading-none">
          Réponse en temps réel <Activity className="h-4 w-4" />
        </div>
        <div className="leading-none text-muted-foreground">
          Je suis encore un modèle en apprentissage, soyez indulgent
        </div>
      </CardFooter>
    </Card>
  );
}
