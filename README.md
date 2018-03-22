
# Evaluating regression lines

### Learning Objectives

* Understand what is meant by the errors of a regression line
* Understand how to calculate the error at a given point
* Understand how to calculate RSS and why we use it as a metric to evaluate a regression line

### Introduction

So far we have seen how to use lines, and formulas for lines to estimate outputs given an input.  We described our lines with two different variables: 

* $m$ -  the slope of the line, and 
* $b$ - the y-intercept

So far we have been rather fast and loose with choosing a line to estimate our output.  Well today, we go further.  In this lesson, we'll calculate how well our regression line matches data we already have.  And from there, will be able improve upon our regression lines such that it matches our data, and thus predicts an output provided an input with more accuracy.

### Determining Quality

To measure the accuracy of a regression line, we see how closely our regression line matches the data we have.  Let's find out what this means.  Below we have data that represents the budget and revenue of four shows, with `x` being the budget and `y` being the revenue.


```python
first_show = {'x': 0, 'y': 100}
second_show = {'x': 100, 'y': 150}
third_show = {'x': 200, 'y': 600}
fourth_show = {'x': 400, 'y': 700}

shows = [first_show, second_show, third_show, fourth_show]
```

> Run code above with shift + enter

For now, let's set a roughshod regression line simply by drawing a line between our first and last points.  Eventually, we'll improve this regression line, so it's ok that we don't use a rigorous technique right now.  We can use our `build_regression_line` function to do so.  You can view the code directly [here](https://github.com/learn-co-curriculum/evaluating-regression-lines/blob/master/linear_equations.py).  


```python
from linear_equations import build_regression_line
x_values = list(map(lambda show: show['x'],shows))
y_values = list(map(lambda show: show['y'],shows))
regression_line = build_regression_line(x_values, y_values)
regression_line
```




    {'b': 100.0, 'm': 1.5}



We can plot our regression line as the following using the [plotting functions](https://github.com/learn-co-curriculum/evaluating-regression-lines/blob/master/graph.py) that we wrote previously:


```python
from graph import m_b_trace, plot, trace_values
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)
data_trace = trace_values(x_values, y_values)
regression_trace = m_b_trace(regression_line['m'], regression_line['b'], x_values)
plot([regression_trace, data_trace])
```


<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>



<div id="8c46ea3e-fdb4-4eb1-a5e0-e4e885201be9" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("8c46ea3e-fdb4-4eb1-a5e0-e4e885201be9", [{"x": [0, 100, 200, 400], "y": [100.0, 250.0, 400.0, 700.0], "mode": "line", "name": "line function"}, {"x": [0, 100, 200, 400], "y": [100, 150, 600, 700], "mode": "markers", "name": "data", "text": []}], {}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>


So that is what our regression line looks like.  And this the line translated into a function.


```python
def sample_regression_formula(x):
    return 1.5(x) + 100
```

Ok so now that we see what our regression line looks like, let's highlight how well our regression line matches our data.

![](./regression-scatter.png)

This is what the above chart is displaying.  That first red line is showing that our regression formula does not perfectly predict our data.  More concretely, at that spot, $x = 100$, our regression line is predicting that the value of $y$ will be 250.  However the point below our regression line shows the actual value of $y$ when $x = 100$ which is 150.  So our regression line is off. 

Each point where our regression line's prediction differs from the actual data is called an error.  So that is what the reds lines are indictating -- the distance between our regression line's predicted value and the actual value.  Or in other words, the red lines are displaying the size of each error.

Now let's measure the size of that error so we can represent the size with a number.  We say that our error is the actual value minus the expected value.  So at point $x = 100$, the actual $y$ is 150 where the expected is $250$.  This translates to $150 - 250 = -100$.  

If we did not have a graph to display this, we calculate this error by using our  formula for the regression line as well as our data.  Our regression formula is $y = 1.5x + 100$.  Then setting $x$ equal to 100, we see that the formula predicts $y = 1.5 * 100 + 100 = 250$.  And we have the actual data point of (100, 150).  So actual - expected = 150 -250 = 100.

### Refining our Terms

Now that we have explained how to calculate an error given a regression line and data, let's express this concept with mathematical notation.  

To start, we want to use notation to distinguish between two things: our predicted $y$ values and our actual $y$ values.  So far we have defined our regression function as $y = mx + b$.  Where for a given value of $x$, we can calculate the value of $y$.  However, this is not totally accurate - as our regression line is not calculating the actual value of $y$ but the *expected * value of $y$. So let's indicate this, by changing our regression line formula to look like the following:

$\overline{y} = \overline{m}x + \overline{b}$ 

Those little dashes over the $y$, $m$ and $b$ are called hats.  So our function reads as y-hat equals m-hat times x plus b-hat.  These hats indicate that this formula does not give us the actual value of $y$, but simply our estimated value of $y$.  And that this estimated value of $y$ is based on our estimated values of $m$ and $b$. 
> Note that $x$ is not a predicted value.  Why is this?  Well, we are *providing* a value $x$, for example a movie budget, not predicting it.  So we are *providing* a value of $x$ and asking it to *predict* a value of $y$.  

Now remember that we were given some real data as well.  This means that we do have actual points for $x$ and $y$, which looks like the following.


```python
first_show = {'x': 0, 'y': 100}
second_show = {'x': 100, 'y': 150}
third_show = {'x': 200, 'y': 600}
fourth_show = {'x': 400, 'y': 700}

shows = [first_show, second_show, third_show, fourth_show]
```

So how do we represent our actual values of $y$? Here's how: $y$.  No extra ink is needed.

Ok, so now we know the following:  
 * **$y$**: actual y  
 * **$\overline{y}$**: estimated y
 
Now, using the Greek letter $\varepsilon$, epsilon, to indicate error, we can say $\varepsilon = y - \overline{y}$.  And, we can be a little more precise by saying we are talking about error at any specific point, where $y$ and $\overline{y}$ are at that $x$ value.  This is written as: 

$\varepsilon _{i}$ = $y_{i}$ - $\overline{y}_{i}$

Now, applying this to a specific point of say when $ x = 100 $, we can say  $\varepsilon _{x=100} = y_{x=100}$ - $\overline{y}_{x=100} = 150 - 250 = 150$

### Calculating and representing total error

Now so far, we know how to calculate the error at a given value of $x$, $x_i$, by using the formula, $\varepsilon_i$ = $y_i - \overline{y_i}$.  And this helpful at describing how well our regression line predicts the value of $y$ at a specific point.  

However, we want to see well our regression describes our dataset in general - not just at a given point.  So let's move beyond calculating the error at a given point to describing the total error of the regression line across all of our data.  As an initial approach, we simply calculate the total error by summing the errors, $y - \overline{y}$, for every point in our dataset.  

Total Error = $\sum_{i=1}^{n} y_i - \overline{y_i}$

This isn't bad, but we'll need to modify this approach slightly. To understand why, let's take another look at our data.

![](./regression-scatter.png)

Take a look at what happens if we add the errors at $x = 100$ and $x = 200$. 

* $\varepsilon_{x=100}= 150 - 250 = -100$
* $\varepsilon_{x=200} = 600 - 400 = 200$  
* $\varepsilon_{x=100} + \varepsilon_{x=200} =  -150 + 200 = 50 $

So because $\varepsilon_{x=100}$ is positive while $ \varepsilon_{x=200} $ is negative, adding the two errors  begins to cancel them out.  That's not what we want.  To prevent this from happening, we can simply square the errors, which ensures that we are always summing positive numbers.

${\varepsilon_i^2}$ = $({y_i - \overline{y_i}})^2$

Now given a list of points with coordinates (x, y), we can calculate the squared error of each of the points, and sum them up.  This is called our ** residual sum of squares ** (RSS).  Using our sigma notation, our formula RSS looks like: 

RSS $ = \sum_{i = 1}^n ({y_i - \overline{y_i}})^2$



So let's apply this to our example.  In our example, we have actual $x$ and $y$ values at the following points: $ (0, 100), (100, 150), (200, 600), (400, 700) $.  And we can calculate the values of $\overline{y} $ as $\overline{y} = 1.5 *x + 100 $, for each of those four points.  So this gives us:

RSS = $(0 - 0)^2 + (150 - 250)^2 + (600 - 400)^2 + (700 - 700)^2$ 

which reduces to  

$-100^2 + 200^2 = 50,000$

Ok, this is great.  So now we have one number, RSS, that does a good job of representing how well our regression line fits the data.  We got there by calculating the errors at each of our provided points, and then squaring the errors so that our errors are always positive.

### Summary 

Previous to this lesson, we have simply assumed that our regression lines make good predictions of $y$ for given values of $x$.  In this lesson, we aimed to find a metric to tell us how well our regression line fits our actual data.  To do this, we started looking at an error at a given point, and defined error as the actual value of $y$ minus the expected value of $y$ from our regression line.  Then we were able to describe how well our regression line describes the entire dataset by squaring the errors at each point (to eliminate negative errors), and adding these errors.  This is called the Residual Sum of Squares (RSS).  This is our metric for describing how well our regression line fits our data. 


```python

```
